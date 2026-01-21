from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import time
import asyncio
import random
import logging
import os
import pickle
import hashlib
import uuid

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()

# --- HEALTH CHECK ---
@app.get("/")
@app.head("/")
async def root():
    return {"status": "online"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 0. CONFIG / MODELS
# ==========================================

class OptimizationRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    initial_capital: float
    min_trades: int
    max_trades: int = 1000
    min_profit_factor: float
    max_drawdown: float = 100.0
    window_days_candidates: List[int] = Field(default=[30, 60, 90, 180])

# ==========================================
# 0.1 FREE-STABLE SETTINGS
# ==========================================

CACHE_DIR = "./.cache"
RESULT_DIR = "./.results"

# Warm plans: (period, interval, refresh_every_seconds)
WARM_PLANS = [
    ("5d",   "1h", 15 * 60),
    ("60d",  "1h", 3 * 60 * 60),
    ("730d", "1h", 12 * 60 * 60),
]

# IMPORTANT: Public endpoints must not trigger upstream fetches.
SERVE_ONLY_FROM_CACHE = True

RESULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours

MAX_CONCURRENT_OPTIMIZATIONS = 2
JOB_TTL_SECONDS = 24 * 60 * 60

RATE_LIMITS = {
    "/asset-stats": (10, 3),
    "/generate-strategy": (60, 2),
    "/job": (2, 15),
    "/activate-symbol": (5, 30),  # allow bursty UI activation
}

# ==========================================
# 0.1.1 UNIVERSAL ASSET WARMING
# ==========================================

KNOWN_TICKERS: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "CAD=X", "AUDUSD=X", "CHF=X", "NZDUSD=X",
    "GC=F", "SI=F", "CL=F", "HG=F",
    "ES=F", "NQ=F", "YM=F", "RTY=F",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "GOOGL", "META",
]

def normalize_symbol(sym: str) -> str:
    return (sym or "").strip()

class WarmRegistry:
    def __init__(self, initial: List[str]):
        self._set = set(normalize_symbol(s) for s in initial)
        self._lock = asyncio.Lock()

    async def add(self, sym: str):
        s = normalize_symbol(sym)
        if not s:
            return
        async with self._lock:
            self._set.add(s)

    async def snapshot(self) -> List[str]:
        async with self._lock:
            return list(self._set)

warm_registry = WarmRegistry(KNOWN_TICKERS)

# ==========================================
# 0.2 SIMPLE IN-MEMORY RATE LIMIT MIDDLEWARE
# ==========================================

class SimpleRateLimiter:
    def __init__(self):
        self.hits: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()

    def _match_rule(self, path: str) -> Optional[Tuple[str, Tuple[int, int]]]:
        matches = [(pfx, rule) for pfx, rule in RATE_LIMITS.items() if path.startswith(pfx)]
        if not matches:
            return None
        matches.sort(key=lambda x: len(x[0]), reverse=True)
        return matches[0]

    async def allow(self, ip: str, path: str) -> Optional[dict]:
        matched = self._match_rule(path)
        if matched is None:
            return None

        prefix, (window, max_req) = matched
        now = time.time()

        key = f"{ip}:{prefix}:{window}:{max_req}"

        async with self.lock:
            arr = self.hits.get(key, [])
            arr = [t for t in arr if now - t < window]
            if len(arr) >= max_req:
                retry_after = int(window - (now - arr[0])) if arr else window
                return {"detail": "Rate limited. Try again later.", "retry_after": max(1, retry_after)}
            arr.append(now)
            self.hits[key] = arr

        return None

rate_limiter = SimpleRateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method in ("OPTIONS", "HEAD"):
        return await call_next(request)

    xff = request.headers.get("x-forwarded-for")
    ip = (xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown"))

    if ip in ("127.0.0.1", "localhost", "::1"):
        return await call_next(request)

    err = await rate_limiter.allow(ip, request.url.path)
    if err:
        return JSONResponse(status_code=429, content=err, headers={"Retry-After": str(err["retry_after"])})

    return await call_next(request)

# ==========================================
# 1. DISK-BASED DATA MANAGER
# ==========================================

class DataManager:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_filename(self, key: str) -> str:
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{self._cache_filename(key)}.pkl")

    def load_from_disk(self, key: str, max_age_seconds: int) -> Optional[pd.DataFrame]:
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            file_age = time.time() - os.path.getmtime(path)
            if file_age <= max_age_seconds:
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {path}: {e}")
        return None

    def load_any_age(self, key: str) -> Optional[pd.DataFrame]:
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {path}: {e}")
            return None

    def save_to_disk(self, key: str, df: pd.DataFrame):
        path = self._get_cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.error(f"Failed to save cache {path}: {e}")

    def _fetch_internal(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        dat = yf.Ticker(ticker)
        df = dat.history(period=period, interval=interval, auto_adjust=True)
        return df

    async def refresh_data(self, ticker: str, period: str, interval: str) -> bool:
        ticker = normalize_symbol(ticker)
        key = f"{ticker}_{period}_{interval}"

        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            await asyncio.sleep(random.uniform(0.25, 0.9))

            for i in range(3):
                try:
                    df = await asyncio.to_thread(self._fetch_internal, ticker, period, interval)
                    if df is None or df.empty:
                        raise RuntimeError("Empty dataframe from source.")

                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

                    if len(df) > 5:
                        self.save_to_disk(key, df)
                        logger.info(f"[WARM] Updated cache {key} ({len(df)} bars)")
                        return True

                except Exception as e:
                    msg = str(e)
                    logger.warning(f"[WARM] Fetch failed {key} attempt {i+1}: {msg}")

                    if "429" in msg or "Too Many Requests" in msg:
                        await asyncio.sleep(90)
                        break

                    await asyncio.sleep(2 ** (i + 1))

        return False

    async def get_cached(self, ticker: str, period: str, interval: str, max_age_seconds: int) -> pd.DataFrame:
        ticker = normalize_symbol(ticker)
        key = f"{ticker}_{period}_{interval}"

        df = self.load_from_disk(key, max_age_seconds=max_age_seconds)
        if df is not None:
            return df

        if SERVE_ONLY_FROM_CACHE:
            df_any = self.load_any_age(key)
            return df_any if df_any is not None else pd.DataFrame()

        ok = await self.refresh_data(ticker, period, interval)
        if ok:
            df2 = self.load_any_age(key)
            return df2 if df2 is not None else pd.DataFrame()

        return pd.DataFrame()

data_manager = DataManager()

# ==========================================
# 1.1 RESULT CACHE + JOB STORE
# ==========================================

os.makedirs(RESULT_DIR, exist_ok=True)

def _result_path(key: str) -> str:
    return os.path.join(RESULT_DIR, f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.pkl")

def load_result_cache(key: str, ttl: int) -> Optional[dict]:
    path = _result_path(key)
    if not os.path.exists(path):
        return None
    try:
        age = time.time() - os.path.getmtime(path)
        if age <= ttl:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.warning(f"Result cache load failed: {e}")
    return None

def save_result_cache(key: str, payload: dict):
    path = _result_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f)
    except Exception as e:
        logger.warning(f"Result cache save failed: {e}")

def request_hash(req: OptimizationRequest) -> str:
    norm = {
        "symbol": normalize_symbol(req.symbol),
        "timeframe": (req.timeframe or "").strip(),
        "initial_capital": float(req.initial_capital),
        "min_trades": int(req.min_trades),
        "max_trades": int(req.max_trades),
        "min_profit_factor": float(req.min_profit_factor),
        "max_drawdown": float(req.max_drawdown),
        "window_days_candidates": list(map(int, req.window_days_candidates)),
    }
    blob = repr(norm).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

jobs: Dict[str, Dict[str, Any]] = {}
hash_to_jobid: Dict[str, str] = {}
jobs_lock = asyncio.Lock()
opt_sema = asyncio.Semaphore(MAX_CONCURRENT_OPTIMIZATIONS)

async def cleanup_jobs_loop():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        async with jobs_lock:
            dead = []
            for jid, j in jobs.items():
                if now - j.get("created_at", now) > JOB_TTL_SECONDS:
                    dead.append(jid)
            for jid in dead:
                h = jobs[jid].get("req_hash")
                if h and hash_to_jobid.get(h) == jid:
                    del hash_to_jobid[h]
                del jobs[jid]

# ==========================================
# 2. MATH HELPERS
# ==========================================

def ensure_1d_len(x, n: int, fill_value=0.0) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    else:
        x = np.asarray(x)
    if x.ndim > 1:
        x = x.flatten()
    x = np.nan_to_num(x.astype(float), nan=fill_value)
    if len(x) > n:
        x = x[:n]
    elif len(x) < n:
        x = np.pad(x, (0, n - len(x)), constant_values=fill_value)
    return x

def calc_sma(series, p): return series.rolling(p).mean()
def calc_ema(series, p): return series.ewm(span=p, adjust=False).mean()

def calc_rsi(series, p):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calc_atr(df, p):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    return tr.rolling(p).mean()

def calc_supertrend(df, period, multiplier):
    high, low, close = df["High"], df["Low"], df["Close"]
    hl2 = (high + low) / 2.0
    atr = calc_atr(df, period)
    up = hl2 + (multiplier * atr)
    dn = hl2 - (multiplier * atr)

    up_val = up.values
    dn_val = dn.values
    close_val = close.values
    m = len(close)
    trend = np.ones(m)
    final_upper = np.zeros(m)
    final_lower = np.zeros(m)

    final_upper[0] = up_val[0] if np.isfinite(up_val[0]) else 0.0
    final_lower[0] = dn_val[0] if np.isfinite(dn_val[0]) else 0.0

    for i in range(1, m):
        prev_fu = final_upper[i-1] if np.isfinite(final_upper[i-1]) else up_val[i-1]
        prev_fl = final_lower[i-1] if np.isfinite(final_lower[i-1]) else dn_val[i-1]

        if close_val[i] > prev_fu:
            trend[i] = 1
        elif close_val[i] < prev_fl:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

        if trend[i] == 1:
            final_lower[i] = max(dn_val[i], prev_fl) if trend[i-1] == 1 else dn_val[i]
            final_upper[i] = np.nan
        else:
            final_upper[i] = min(up_val[i], prev_fu) if trend[i-1] == -1 else up_val[i]
            final_lower[i] = np.nan

    return pd.Series(trend, index=df.index).fillna(1).to_numpy()

def calc_bollinger_bands(series, p, std_dev):
    sma = series.rolling(p).mean()
    std = series.rolling(p).std()
    return sma + (std * std_dev), sma - (std * std_dev)

def calc_adx(df, p):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    atr = tr.rolling(p).mean()
    up, down = high.diff(), -low.diff()
    pos = np.where((up > down) & (up > 0), up, 0.0)
    neg = np.where((down > up) & (down > 0), down, 0.0)
    pos_s = pd.Series(pos, index=df.index).rolling(p).mean()
    neg_s = pd.Series(neg, index=df.index).rolling(p).mean()
    pos_di = 100 * (pos_s / atr)
    neg_di = 100 * (neg_s / atr)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return dx.rolling(p).mean().fillna(0)

def calc_vwap(df):
    v = df["Volume"]
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    denom = v.cumsum().replace(0, np.nan)
    return ((tp * v).cumsum() / denom).ffill().fillna(0.0)

def calc_ichimoku(df, t, k, s):
    high, low = df["High"], df["Low"]
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2.0
    kijun = (high.rolling(k).max() + low.rolling(k).min()) / 2.0
    ssa = ((tenkan + kijun) / 2.0).shift(k)
    ssb = ((high.rolling(s).max() + low.rolling(s).min()) / 2.0).shift(k)
    top = np.maximum(ssa, ssb)
    bot = np.minimum(ssa, ssb)
    return np.where(df["Close"] > top, 1, np.where(df["Close"] < bot, -1, 0))

def calc_zscore(series, p):
    mean = series.rolling(p).mean()
    std = series.rolling(p).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0)

def calc_tema(series, p):
    ema1 = series.ewm(span=p, adjust=False).mean()
    ema2 = ema1.ewm(span=p, adjust=False).mean()
    ema3 = ema2.ewm(span=p, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

# ==========================================
# 3. PINE SCRIPT GENERATOR (unchanged)
# ==========================================

def get_pine_script(name, params, strategy_type, symbol):
    script = f"""//@version=5
strategy("Lux AI: {name} [{symbol}]", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_value=0.04)
"""
    logic = ""

    if strategy_type == "Connors_Pullback":
        script += f"maLen = {params[0]}\nrsiLen = {params[1]}\nlimit = {params[2]}\n"
        logic = "ma = ta.sma(close, maLen)\nrsiVal = ta.rsi(close, rsiLen)\nlongCondition = close > ma and rsiVal < limit\nshortCondition = close < ma and rsiVal > (100 - limit)"
    elif strategy_type == "Impulse_MACD":
        script += f"vol_mult = {params[0]}\n"
        logic = "[macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)\nvolAvg = ta.sma(volume, 20)\nimpulse = volume > volAvg * vol_mult\nlongCondition = ta.crossover(macdLine, signalLine) and impulse\nshortCondition = ta.crossunder(macdLine, signalLine) and impulse"
    elif strategy_type == "VWAP_Reversion":
        script += f"dev = {params[0]}\n"
        logic = "vwapVal = ta.vwap(close)\nstdev = ta.stdev(close, 20)\nupper = vwapVal + (stdev * dev)\nlower = vwapVal - (stdev * dev)\nlongCondition = close < lower\nshortCondition = close > upper"
    elif strategy_type == "Regime_Trend":
        script += f"fastLen = {params[0]}\nslowLen = {params[1]}\n"
        logic = "fast = ta.ema(close, fastLen)\nslow = ta.ema(close, slowLen)\n[_, _, adx] = ta.dmi(14, 14)\nlongCondition = ta.crossover(fast, slow) and adx > 20\nshortCondition = ta.crossunder(fast, slow) and adx > 20"
    elif strategy_type == "RSI_Bollinger":
        script += f"len = {params[0]}\nthresh = {params[1]}\nbb_len = 20\nbb_mult = 2.0\n"
        logic = "rsiVal = ta.rsi(close, len)\n[_, upper, lower] = ta.bb(close, bb_len, bb_mult)\nlongCondition = rsiVal < thresh and close < lower\nshortCondition = rsiVal > (100 - thresh) and close > upper"
    elif strategy_type == "SuperTrend_Vol":
        script += f"atrPeriod = {params[0]}\nfactor = {params[1]}\n"
        logic = "[_, direction] = ta.supertrend(factor, atrPeriod)\nlongCondition = direction < 0\nshortCondition = direction > 0"
    elif strategy_type == "Ichimoku":
        script += f"t = {params[0]}\nk = {params[1]}\ns = {params[2]}\n"
        logic = "tenkan = (ta.highest(high, t) + ta.lowest(low, t)) / 2\nkijun = (ta.highest(high, k) + ta.lowest(low, k)) / 2\nleadA = ((tenkan + kijun) / 2)[k]\nlongCondition = close > leadA\nshortCondition = close < leadA"
    elif strategy_type == "Z-Score":
        script += f"len = {params[0]}\nthresh = {params[1]}\n"
        logic = "basis = ta.sma(close, len)\ndev = ta.stdev(close, len)\nz = (close - basis) / dev\nlongCondition = z < -thresh\nshortCondition = z > thresh"
    elif strategy_type == "Engulfing":
        logic = "bull = close > open and close[1] < open[1] and close > open[1] and open < close[1]\nbear = close < open and close[1] > open[1] and close < open[1] and open > close[1]\nlongCondition = bull\nshortCondition = bear"
    elif strategy_type == "TEMA_Cross":
        script += f"f = {params[0]}\ns = {params[1]}\n"
        logic = "tema1 = ta.tema(close, f)\ntema2 = ta.tema(close, s)\nlongCondition = ta.crossover(tema1, tema2)\nshortCondition = ta.crossunder(tema1, tema2)"
    elif strategy_type == "Williams_R":
        script += f"len = {params[0]}\nemaLen = {params[1]}\n"
        logic = "wr = ta.wpr(len)\nema = ta.ema(close, emaLen)\nlongCondition = wr < -80 and close > ema\nshortCondition = wr > -20 and close < ema"

    script += f"""
{logic}

if (longCondition)
    strategy.entry("Long", strategy.long, comment="ALPHA_BUY")
if (shortCondition)
    strategy.entry("Short", strategy.short, comment="ALPHA_SELL")

plotshape(longCondition, style=shape.labelup, location=location.belowbar, color=color.green, size=size.tiny, title="Buy")
plotshape(shortCondition, style=shape.labeldown, location=location.abovebar, color=color.red, size=size.tiny, title="Sell")
"""
    return script

# ==========================================
# 4. BACKTEST ENGINE
# ==========================================

def compute_risk_managed_series(df, signals, capital):
    n = len(df)
    sig = ensure_1d_len(signals, n)

    pos_arr = np.roll(sig, 1)
    pos_arr[0] = 0.0

    close_pct = df["Close"].pct_change().fillna(0.0).to_numpy()

    leveraged_return = (pos_arr * close_pct) * 3.0
    equity_arr = capital * np.cumprod(1 + leveraged_return)
    trade_diff = np.abs(np.diff(pos_arr, prepend=0))
    return leveraged_return, equity_arr, trade_diff

def evaluate_window(ret_slice, equity_slice, trade_diff_slice, dates, initial_cap):
    if len(ret_slice) < 2:
        return 0, 0, 0, 0, 0, 0, "", ""

    final_eq = float(equity_slice[-1])
    window_start_eq = float(equity_slice[0])
    pnl_pct = ((final_eq - window_start_eq) / window_start_eq) * 100

    wins = np.sum(ret_slice[ret_slice > 0])
    losses = np.abs(np.sum(ret_slice[ret_slice < 0]))
    pf = float(wins / losses) if losses > 0 else (5.0 if wins > 0 else 0.0)

    std_dev = np.std(ret_slice)
    sharpe = 0.0
    if std_dev > 0:
        sharpe = (np.mean(ret_slice) / std_dev) * np.sqrt(len(ret_slice))

    peak = np.maximum.accumulate(equity_slice)
    dd = (equity_slice - peak) / (peak + 1e-9)
    max_dd = float(np.abs(np.min(dd)) * 100)

    txns = int(np.sum(trade_diff_slice) / 2)
    start_date = dates[0].strftime("%Y-%m-%d")
    end_date = dates[-1].strftime("%Y-%m-%d")

    return final_eq, txns, pf, pnl_pct, max_dd, sharpe, start_date, end_date

# ==========================================
# 5. WARM CACHE BACKGROUND TASKS
# ==========================================

async def warm_cache_loop():
    next_run: Dict[tuple, float] = {plan: 0.0 for plan in WARM_PLANS}

    while True:
        now = time.time()
        symbols = await warm_registry.snapshot()

        for (period, interval, every_s) in WARM_PLANS:
            plan_key = (period, interval, every_s)
            if now >= next_run[plan_key]:
                for sym in symbols:
                    await data_manager.refresh_data(sym, period, interval)

                next_run[plan_key] = now + every_s + random.uniform(3, 15)

        await asyncio.sleep(5)

@app.on_event("startup")
async def on_startup():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    asyncio.create_task(warm_cache_loop())
    asyncio.create_task(cleanup_jobs_loop())
    logger.info("Startup complete: warm cache + job cleanup running.")

# ==========================================
# 6. ENDPOINTS
# ==========================================

@app.post("/activate-symbol")
async def activate_symbol(symbol: str):
    await warm_registry.add(symbol)
    return {"ok": True, "symbol": symbol}

@app.get("/asset-stats")
async def get_asset_stats(symbol: str):
    try:
        ticker = normalize_symbol(symbol)
        await warm_registry.add(ticker)

        df = await data_manager.get_cached(ticker, "5d", "1h", max_age_seconds=30 * 60)
        if df.empty or "Close" not in df.columns:
            return {"price": 0, "change": 0, "volatility": 0, "stale": True}

        close = df["Close"]
        if len(close) < 2:
            return {"price": float(close.iloc[-1]) if len(close) else 0, "change": 0, "volatility": 0, "stale": True}

        current = float(close.iloc[-1])
        start = float(close.iloc[0]) if float(close.iloc[0]) != 0 else current
        change = ((current - start) / start) * 100 if start != 0 else 0.0
        vol = float(close.pct_change().std() * 100) * 4

        key = f"{ticker}_5d_1h"
        path = data_manager._get_cache_path(key)
        stale = True
        try:
            stale = (time.time() - os.path.getmtime(path)) > (30 * 60)
        except Exception:
            pass

        return {"price": current, "change": round(change, 2), "volatility": round(vol, 2), "stale": stale}
    except Exception as e:
        logger.error(f"Asset Stats Error: {e}")
        return {"price": 0, "change": 0, "volatility": 0, "stale": True}

@app.post("/generate-strategy")
async def generate_strategy(req: OptimizationRequest):
    await warm_registry.add(req.symbol)

    h = request_hash(req)
    cached = load_result_cache(h, RESULT_TTL_SECONDS)
    if cached is not None:
        return {"status": "done", "job_id": None, "cached": True, "result": cached}

    async with jobs_lock:
        existing = hash_to_jobid.get(h)
        if existing and existing in jobs and jobs[existing]["status"] in ("queued", "running"):
            return {"status": jobs[existing]["status"], "job_id": existing, "cached": False}

        job_id = uuid.uuid4().hex
        jobs[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "req_hash": h,
            "result": None,
            "error": None,
        }
        hash_to_jobid[h] = job_id

    asyncio.create_task(_run_optimization_job(job_id, req))
    return {"status": "queued", "job_id": job_id, "cached": False}

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    async with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job_id,
            "status": j["status"],
            "error": j["error"],
            "result": j["result"] if j["status"] == "done" else None,
        }

# ==========================================
# 7. OPTIMIZATION WORKER
# ==========================================

async def _run_optimization_job(job_id: str, req: OptimizationRequest):
    async with opt_sema:
        async with jobs_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "running"

        try:
            payload = await _compute_strategy(req)
            h = jobs[job_id]["req_hash"]
            save_result_cache(h, payload)

            async with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["result"] = payload
                    jobs[job_id]["error"] = None

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            async with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = str(e)

def _bars_per_day_from_timeframe(tf: str) -> int:
    tf = (tf or "").strip().lower()
    if tf.endswith("h"):
        try:
            h = int(tf[:-1])
            return max(1, int(24 / h))
        except:
            return 24
    if tf.endswith("d"):
        return 1
    return 24

async def _compute_strategy(req: OptimizationRequest) -> dict:
    logger.info(f"--> QUANTUM SCAN (JOB): {req.symbol}...")

    ticker = normalize_symbol(req.symbol)

    df = await data_manager.get_cached(ticker, "730d", req.timeframe, max_age_seconds=24 * 60 * 60)
    if df.empty:
        df = await data_manager.get_cached(ticker, "60d", req.timeframe, max_age_seconds=6 * 60 * 60)

    if df.empty:
        raise HTTPException(status_code=503, detail="Data not available yet. Please retry shortly.")

    # Minimal column sanity
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        raise HTTPException(status_code=500, detail=f"Data missing columns: {sorted(list(needed - set(df.columns)))}")

    n = len(df)
    bars_per_day = _bars_per_day_from_timeframe(req.timeframe)

    available_days = max(1, n // max(1, bars_per_day))

    # FIX: ALWAYS have at least one window, even for small datasets.
    scan_windows = [w for w in req.window_days_candidates if (w * bars_per_day) < n]
    if not scan_windows:
        # choose the largest sensible window we can actually scan
        fallback_days = max(2, min(available_days, 30))
        scan_windows = [fallback_days]

    pool = []

    # Create a usable "Volume" for markets where Yahoo returns 0 volume (forex often).
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    else:
        if float(df["Volume"].sum()) == 0.0:
            df["Volume"] = 1.0

    # 1) RSI + BB
    upper, lower = calc_bollinger_bands(df["Close"], 20, 2.0)
    for rsi_len in [9, 14, 21]:
        rsi = calc_rsi(df["Close"], rsi_len)
        sig = np.where((rsi < 30) & (df["Close"] < lower), 1,
                       np.where((rsi > 70) & (df["Close"] > upper), -1, 0))
        pool.append({"type": "RSI_Bollinger", "args": (rsi_len, 30), "sig": sig, "tag": f"Reversion ({rsi_len})"})

    # 2) Supertrend + volume filter
    vol_ma = df["Volume"].rolling(20).mean()
    high_vol = df["Volume"] > (vol_ma * 1.2)
    for factor in [2, 3]:
        st_sig = calc_supertrend(df, 10, factor)
        sig = np.where((st_sig == 1) & high_vol, 1,
                       np.where((st_sig == -1) & high_vol, -1, 0))
        pool.append({"type": "SuperTrend_Vol", "args": (10, factor), "sig": sig, "tag": f"Trend+Vol ({factor})"})

    # 3) EMA regime trend
    adx = calc_adx(df, 14)
    trending = adx > 20
    fast = calc_ema(df["Close"], 9)
    slow = calc_ema(df["Close"], 21)
    pool.append({"type": "Regime_Trend", "args": (9, 21),
                 "sig": np.where((fast > slow) & trending, 1, np.where((fast < slow) & trending, -1, 0)),
                 "tag": "Smart Trend"})

    # 4) Ichimoku
    pool.append({"type": "Ichimoku", "args": (9, 26, 52), "sig": calc_ichimoku(df, 9, 26, 52), "tag": "Classic Trend"})

    # 5) Z-score
    for p, t in itertools.product([20], [2.0]):
        z = calc_zscore(df["Close"], p)
        pool.append({"type": "Z-Score", "args": (p, t), "sig": np.where(z < -t, 1, np.where(z > t, -1, 0)), "tag": "Statistical"})

    # 6) Engulfing (price action)
    o = df["Open"].values
    c = df["Close"].values
    po = np.roll(o, 1)
    pc = np.roll(c, 1)
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    pool.append({"type": "Engulfing", "args": (), "sig": np.where(bull, 1, np.where(bear, -1, 0)), "tag": "Price Action"})

    # 7) TEMA cross
    for f, s in [(9, 21), (21, 50)]:
        fast_t, slow_t = calc_tema(df["Close"], f), calc_tema(df["Close"], s)
        pool.append({"type": "TEMA_Cross", "args": (f, s), "sig": np.where(fast_t > slow_t, 1, -1), "tag": "Scalping"})

    # 8) Williams R / Connors if enough data
    if n > 250:
        hh = df["High"].rolling(14).max()
        ll = df["Low"].rolling(14).min()
        wr = (-100 * (hh - df["Close"]) / (hh - ll)).fillna(-50)
        ema50 = calc_ema(df["Close"], 50)
        pool.append({"type": "Williams_R", "args": (14, 50),
                     "sig": np.where((wr < -80) & (df["Close"] > ema50), 1,
                                     np.where((wr > -20) & (df["Close"] < ema50), -1, 0)),
                     "tag": "Momentum"})

        sma200 = calc_sma(df["Close"], 200)
        rsi2 = calc_rsi(df["Close"], 2)
        pool.append({"type": "Connors_Pullback", "args": (200, 2, 5),
                     "sig": np.where((df["Close"] > sma200) & (rsi2 < 5), 1,
                                     np.where((df["Close"] < sma200) & (rsi2 > 95), -1, 0)),
                     "tag": "Inst. Pullback"})

    # 9) VWAP mean reversion
    vwap = calc_vwap(df)
    std = df["Close"].rolling(20).std()
    pool.append({"type": "VWAP_Reversion", "args": (2.0,),
                 "sig": np.where(df["Close"] < vwap - (2 * std), 1, np.where(df["Close"] > vwap + (2 * std), -1, 0)),
                 "tag": "Mean Reversion"})

    # 10) MACD impulse
    ema12 = calc_ema(df["Close"], 12)
    ema26 = calc_ema(df["Close"], 26)
    macd = ema12 - ema26
    signal = calc_ema(macd, 9)
    high_vol_2 = df["Volume"] > (vol_ma * 1.5)
    pool.append({"type": "Impulse_MACD", "args": (1.5,),
                 "sig": np.where((macd > signal) & high_vol_2, 1, np.where((macd < signal) & high_vol_2, -1, 0)),
                 "tag": "Volume Momentum"})

    valid_candidates, fallback_candidates = [], []
    winner = None

    for s in pool:
        ret, equity, t_diff = compute_risk_managed_series(df, s["sig"], req.initial_capital)

        for days in sorted(scan_windows, reverse=True):
            w_len = int(days * bars_per_day)
            if w_len >= n:
                continue

            start = n - w_len
            f, txns, pf, pnl, dd, sharpe, s_date, e_date = evaluate_window(
                ret[start:], equity[start:], t_diff[start:], df.index[start:], req.initial_capital
            )

            curve_slice = equity[start:]
            if len(curve_slice) < 2:
                continue

            rebased_curve = (curve_slice / curve_slice[0]) * req.initial_capital

            cand = {
                "name": f"{s['tag']}: {s['type']}",
                "days": days,
                "txns": txns,
                "pf": pf,
                "pnl": pnl,
                "dd": dd,
                "final": f,
                "sharpe": sharpe,
                "pine": get_pine_script(s["type"], s["args"], s["type"], req.symbol),
                "tag": s["tag"],
                "curve": rebased_curve,
                "start": start,
                "start_date": s_date,
                "end_date": e_date,
            }

            if txns >= req.min_trades and pf >= req.min_profit_factor and dd <= req.max_drawdown:
                valid_candidates.append(cand)
            else:
                v_score = min(txns / max(req.min_trades, 1), 1.5)
                cand["score"] = (pf * v_score) + (sharpe * 0.5) - (dd / 50)
                fallback_candidates.append(cand)

    if valid_candidates:
        valid_candidates.sort(key=lambda x: -x["sharpe"])
        winner, note = valid_candidates[0], f"High Sharpe ({valid_candidates[0]['sharpe']:.2f})"
    elif fallback_candidates:
        fallback_candidates.sort(key=lambda x: -x["score"])
        winner, note = fallback_candidates[0], f"Best Fallback ({fallback_candidates[0]['txns']} Trades)"
    else:
        raise HTTPException(status_code=500, detail="No candidates produced (insufficient data / invalid windows).")

    # ==========================================
    # SMOOTHING APPLIED HERE
    # ==========================================
    raw_curve = winner["curve"]

    # Use a rolling mean to smooth out the steps
    window_size = max(5, int(len(raw_curve) * 0.04))
    smoothed_series = pd.Series(raw_curve).rolling(window=window_size, min_periods=1, center=True).mean()
    subset_vals = smoothed_series.to_numpy()

    subset_idx = df.index[winner["start"]:]

    # Reduce downsampling to keep the curve smooth visually
    downsample = 3 if len(subset_vals) > 2000 else 1

    chart_data = [
        {"time": int(t.timestamp()), "value": float(v)}
        for t, v in zip(subset_idx[::downsample], subset_vals[::downsample])
    ]

    return {
        "strategy_name": winner["name"],
        "window_days": winner["days"],
        "metrics": {
            "total_pnl": winner["final"] - req.initial_capital,
            "pnl_percent": winner["pnl"],
            "total_txns": winner["txns"],
            "profit_factor": round(winner["pf"], 2),
            "pine_code": winner["pine"],
            "max_dd": round(winner["dd"], 2),
            "tag": winner["tag"],
            "duration": f"{winner['days']} Days",
            "note": note,
            "start_date": winner["start_date"],
            "end_date": winner["end_date"],
        },
        "chart_data": chart_data,
        "data_source": "cache_only",
    }
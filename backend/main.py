from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import time

app = FastAPI()

# --- HEALTH CHECK ENDPOINT (FIXES 404 ERRORS) ---
@app.get("/")
@app.head("/")
async def root():
    return {"status": "online"}
# ------------------------------------------------

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
    window_mode: str = "auto_shortest"
    window_days_candidates: List[int] = Field(default=[14, 21, 30, 45, 60, 90, 120, 180])
    stability_windows: int = 1
    min_window_days: int = 7


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def ensure_1d_len(x, n: int, index=None, fill_value=0.0) -> np.ndarray:
    if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        if index is not None: x = x.reindex(index)
        x = x.to_numpy()
    else: x = np.asarray(x)
    if x.ndim > 1: x = x[:, 0]
    try: x = x.astype(float, copy=False)
    except Exception: x = x.astype(np.float64)
    if len(x) > n: x = x[:n]
    elif len(x) < n: x = np.pad(x, (0, n - len(x)), constant_values=fill_value)
    x = np.nan_to_num(x, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return x

def assert_sig(name: str, sig: np.ndarray, n: int):
    if len(sig) != n:
        raise ValueError(f"Signal length mismatch for {name}: {len(sig)} vs {n}")


# ==========================================
# 2. INDICATOR LIBRARY
# ==========================================

def calc_sma(series, p): return series.rolling(p).mean()
def calc_ema(series, p): return series.ewm(span=p, adjust=False).mean()

def calc_tema(series, p):
    ema1 = series.ewm(span=p, adjust=False).mean()
    ema2 = ema1.ewm(span=p, adjust=False).mean()
    ema3 = ema2.ewm(span=p, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def calc_rsi(series, p):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calc_zscore(series, p):
    mean = series.rolling(p).mean()
    std = series.rolling(p).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0)

def calc_obv_slope(df, p):
    close = ensure_1d_len(df["Close"], len(df), df.index)
    volume = ensure_1d_len(df["Volume"], len(df), df.index)
    diff = np.diff(close, prepend=close[0])
    direction = np.sign(diff)
    obv = np.cumsum(direction * volume)
    return pd.Series(obv, index=df.index).diff(p).fillna(0).to_numpy()

def calc_cmf(df, p):
    high, low, close = ensure_1d_len(df["High"], len(df), df.index), ensure_1d_len(df["Low"], len(df), df.index), ensure_1d_len(df["Close"], len(df), df.index)
    volume = ensure_1d_len(df["Volume"], len(df), df.index)
    denom = (high - low)
    denom = np.where(denom == 0, np.nan, denom)
    mfv = ((close - low) - (high - close)) / denom
    mfv = np.nan_to_num(mfv) * volume
    res = pd.Series(mfv, index=df.index).rolling(p).sum() / pd.Series(volume, index=df.index).rolling(p).sum()
    return res.fillna(0).to_numpy()

def calc_coppock(df, roc1, roc2, w):
    close = df["Close"]
    roc_l = close.pct_change(roc1)
    roc_s = close.pct_change(roc2)
    return (roc_l + roc_s).ewm(span=w, adjust=False).mean().fillna(0).to_numpy()

def calc_std_error_bands(df, p, dev):
    close = df["Close"]
    lin_reg = close.rolling(p).mean()
    std = close.rolling(p).std()
    return lin_reg + (dev * std), lin_reg - (dev * std)

def calc_supertrend(df, period, multiplier):
    high, low, close = ensure_1d_len(df["High"], len(df), df.index), ensure_1d_len(df["Low"], len(df), df.index), ensure_1d_len(df["Close"], len(df), df.index)
    hl2 = (high + low) / 2.0
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr, index=df.index).rolling(period).mean().to_numpy()
    up = hl2 + (multiplier * atr)
    dn = hl2 - (multiplier * atr)
    prev_up = np.roll(up, 1); prev_up[0] = up[0]
    prev_dn = np.roll(dn, 1); prev_dn[0] = dn[0]
    sig = np.where(close > prev_up, 1, np.where(close < prev_dn, -1, 0))
    return pd.Series(sig, index=df.index).replace(0, np.nan).ffill().fillna(0).to_numpy()

def calc_adx(df, p):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    tr_s = tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    up, down = high.diff(), -low.diff()
    pos = np.where((up > down) & (up > 0), up, 0.0)
    neg = np.where((down > up) & (down > 0), down, 0.0)
    pos_s = pd.Series(pos, index=df.index).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    neg_s = pd.Series(neg, index=df.index).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    pos_di = 100 * (pos_s / tr_s.replace(0, np.nan))
    neg_di = 100 * (neg_s / tr_s.replace(0, np.nan))
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return dx.ewm(alpha=1/p, adjust=False).mean().fillna(0).to_numpy()

def calc_stoch_rsi(df, p, k, d):
    rsi = calc_rsi(df["Close"], p)
    min_r, max_r = rsi.rolling(p).min(), rsi.rolling(p).max()
    denom = (max_r - min_r).replace(0, np.nan)
    stoch = (rsi - min_r) / denom * 100
    return stoch.rolling(k).mean().fillna(50).to_numpy()

def calc_mfi(df, p):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"]
    diff = tp.diff()
    pos = rmf.where(diff > 0, 0.0).rolling(p).sum()
    neg = rmf.where(diff < 0, 0.0).rolling(p).sum()
    return (100 - (100 / (1 + (pos / neg.replace(0, np.nan))))).fillna(50)

def calc_connors_rsi(df, rsi_p, streak_p, roc_p):
    close = df["Close"]
    rsi = calc_rsi(close, rsi_p)
    streak_rsi = calc_rsi(close.diff().fillna(0), streak_p)
    roc_rank = close.pct_change().rolling(roc_p).apply(lambda x: (x < x.iloc[-1]).sum() / len(x) * 100, raw=False)
    return ((rsi + streak_rsi + roc_rank) / 3.0).fillna(50).to_numpy()

def calc_williams(df, p):
    hh, ll = df["High"].rolling(p).max(), df["Low"].rolling(p).min()
    return (-100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)).fillna(-50).to_numpy()

def calc_aroon(df, p):
    high, low = ensure_1d_len(df["High"], len(df), df.index), ensure_1d_len(df["Low"], len(df), df.index)
    n = len(high)
    aroon_up, aroon_down = np.zeros(n), np.zeros(n)
    for i in range(p, n):
        aroon_up[i] = ((p - (p - 1 - np.argmax(high[i - p + 1:i + 1]))) / p) * 100
        aroon_down[i] = ((p - (p - 1 - np.argmin(low[i - p + 1:i + 1]))) / p) * 100
    return pd.Series(aroon_up - aroon_down, index=df.index).fillna(0).to_numpy()

def calc_vortex(df, p):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    vm_plus, vm_minus = (high - low.shift(1)).abs(), (low - high.shift(1)).abs()
    vi_plus = vm_plus.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)
    vi_minus = vm_minus.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)
    return np.where(vi_plus.fillna(0) > vi_minus.fillna(0), 1, -1)

def calc_donchian(df, p):
    upper = df["High"].rolling(p).max()
    return np.where(df["Close"] > upper.shift(1), 1, -1)

def calc_ichimoku(df, t, k, s):
    high, low = df["High"], df["Low"]
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2.0
    kijun = (high.rolling(k).max() + low.rolling(k).min()) / 2.0
    ssa = ((tenkan + kijun) / 2.0).shift(k)
    ssb = ((high.rolling(s).max() + low.rolling(s).min()) / 2.0).shift(k)
    top, bot = np.maximum(ssa, ssb), np.minimum(ssa, ssb)
    return np.where(df["Close"] > top, 1, np.where(df["Close"] < bot, -1, 0))


# ==========================================
# 3. STRATEGIES
# ==========================================

def strat_engulfing(df):
    o, c = ensure_1d_len(df["Open"], len(df), df.index), ensure_1d_len(df["Close"], len(df), df.index)
    po, pc = np.roll(o, 1), np.roll(c, 1); po[0], pc[0] = o[0], c[0]
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    return np.where(bull, 1, np.where(bear, -1, 0))

def strat_zscore(df, p, t):
    z = calc_zscore(df["Close"], p)
    return np.where(z < -t, 1, np.where(z > t, -1, 0))

def strat_obv(df, p):
    return np.where(calc_obv_slope(df, p) > 0, 1, -1)

def strat_cmf(df, p, l):
    cmf = calc_cmf(df, p)
    return np.where(cmf > l, 1, np.where(cmf < -l, -1, 0))

def strat_tema(df, f, s):
    fast, slow = calc_tema(df["Close"], f), calc_tema(df["Close"], s)
    return np.where(fast > slow, 1, -1)

def strat_coppock(df, r1, r2, w):
    return np.where(calc_coppock(df, r1, r2, w) > 0, 1, -1)

def strat_std_err(df, p, d):
    m, _ = calc_std_error_bands(df, p, d)
    std = df["Close"].rolling(p).std()
    u, l = m + (d * std), m - (d * std)
    return np.where(df["Close"] < l, 1, np.where(df["Close"] > u, -1, 0))

def strat_ema(df, f, s):
    fast, slow = calc_ema(df["Close"], f), calc_ema(df["Close"], s)
    return np.where(fast > slow, 1, -1)

def strat_rsi(df, p, l):
    r = calc_rsi(df["Close"], p)
    return np.where(r < l, 1, np.where(r > 100 - l, -1, 0))

def strat_stoch(df, p, k, d_val, l):
    s = calc_stoch_rsi(df, p, k, d_val)
    return np.where(s < l, 1, np.where(s > 100 - l, -1, 0))

def strat_bb_adx(df, w, d, t):
    ma = df["Close"].rolling(w).mean()
    std = df["Close"].rolling(w).std()
    adx = calc_adx(df, 14)
    return np.where((df["Close"] > ma + d * std) & (adx > t), 1, np.where((df["Close"] < ma - d * std) & (adx > t), -1, 0))

def strat_keltner(df, e, a, m):
    ema = calc_ema(df["Close"], e)
    prev_close = df["Close"].shift(1)
    tr = np.maximum(df["High"] - df["Low"], (df["High"] - prev_close).abs())
    atr = tr.rolling(a).mean()
    return np.where(df["Close"] > ema + m * atr, 1, np.where(df["Close"] < ema - m * atr, -1, 0))

def strat_mfi(df, p, r_p):
    mfi, r = calc_mfi(df, p), calc_rsi(df["Close"], r_p)
    return np.where((mfi < 20) & (r < 30), 1, np.where((mfi > 80) & (r > 70), -1, 0))

def strat_connors(df, r, s, roc):
    c = calc_connors_rsi(df, r, s, roc)
    return np.where(c < 10, 1, np.where(c > 90, -1, 0))

def strat_will(df, p, e_p):
    w, e = calc_williams(df, p), calc_ema(df["Close"], e_p).to_numpy()
    close = df["Close"].to_numpy()
    return np.where((w < -80) & (close > e), 1, np.where((w > -20) & (close < e), -1, 0))

def strat_sqz(df, l, b, k):
    ma = df["Close"].rolling(l).mean(); std = df["Close"].rolling(l).std()
    prev_close = df["Close"].shift(1)
    tr = np.maximum(df["High"] - df["Low"], (df["High"] - prev_close).abs())
    atr = tr.rolling(l).mean()
    sqz = (ma + b * std < ma + k * atr) & (ma - b * std > ma - k * atr)
    mom = df["Close"] > calc_ema(df["Close"], 20)
    return np.where(sqz & mom, 1, np.where(sqz & ~mom, -1, 0))


# ==========================================
# 4. PINE SCRIPT GENERATOR
# ==========================================

def get_pine_script(name, params, invert):
    logic = "// Logic"
    if "Engulfing" in name:
        logic = (
            "bullEng = close > open and close[1] < open[1] and close > open[1] and open < close[1]\n"
            "bearEng = close < open and close[1] > open[1] and close < open[1] and open > close[1]\n"
            "longCondition = bullEng\nshortCondition = bearEng"
        )
    elif "Z-Score" in name:
        logic = f"basis = ta.sma(close, {params[0]})\ndev = ta.stdev(close, {params[0]})\nz = (close - basis) / dev\nlongCondition = z < -{params[1]}\nshortCondition = z > {params[1]}"
    elif "OBV Trend" in name:
        logic = f"obv = ta.obv\nlongCondition = obv > obv[{params[0]}]\nshortCondition = obv < obv[{params[0]}]"
    elif "CMF" in name:
        logic = f"ad = close==high and close==low or high==low ? 0 : ((2*close-low-high)/(high-low))*volume\nmf = ta.sma(ad, {params[0]}) / ta.sma(volume, {params[0]})\nlongCondition = mf > {params[1]}\nshortCondition = mf < -{params[1]}"
    elif "TEMA" in name:
        logic = f"tema1 = ta.tema(close, {params[0]})\ntema2 = ta.tema(close, {params[1]})\nlongCondition = ta.crossover(tema1, tema2)\nshortCondition = ta.crossunder(tema1, tema2)"
    elif "Coppock" in name:
        logic = f"wma(src, len) => ta.wma(src, len)\nlongCondition = ta.wma(ta.roc(close, {params[0]}) + ta.roc(close, {params[1]}), {params[2]}) > 0\nshortCondition = ta.wma(ta.roc(close, {params[0]}) + ta.roc(close, {params[1]}), {params[2]}) < 0"
    elif "Std Error" in name:
        logic = f"basis = ta.linreg(close, {params[0]}, 0)\nlongCondition = close < (basis - {params[1]}*ta.stdev(close, {params[0]}))\nshortCondition = close > (basis + {params[1]}*ta.stdev(close, {params[0]}))"
    elif "EMA" in name:
        logic = f"longCondition = ta.crossover(ta.ema(close, {params[0]}), ta.ema(close, {params[1]}))\nshortCondition = ta.crossunder(ta.ema(close, {params[0]}), ta.ema(close, {params[1]}))"
    elif "RSI" in name:
        logic = f"longCondition = ta.rsi(close, {params[0]}) < {params[1]}\nshortCondition = ta.rsi(close, {params[0]}) > {100 - params[1]}"
    elif "SuperTrend" in name:
        logic = f"longCondition = ta.supertrend({params[1]}, {params[0]})[1] < 0\nshortCondition = ta.supertrend({params[1]}, {params[0]})[1] > 0"
    elif "BB+ADX" in name:
        logic = f"longCondition = close > (ta.sma(close, {params[0]}) + {params[1]}*ta.stdev(close, {params[0]})) and ta.adx(14) > {params[2]}\nshortCondition = close < (ta.sma(close, {params[0]}) - {params[1]}*ta.stdev(close, {params[0]})) and ta.adx(14) > {params[2]}"
    elif "Keltner" in name:
        logic = f"[_, u, l] = ta.kc(close, {params[0]}, {params[2]})\nlongCondition = close > u\nshortCondition = close < l"
    elif "TTM Squeeze" in name:
        logic = f"[_, u, l] = ta.bb(close, {params[0]}, {params[1]})\n[_, ku, kl] = ta.kc(close, {params[0]}, {params[2]})\nlongCondition = (u < ku) and close > ta.ema(close, 20)\nshortCondition = (l > kl) and close < ta.ema(close, 20)"
    elif "Ichimoku" in name:
        logic = f"tenkan = (ta.highest(high, {params[0]}) + ta.lowest(low, {params[0]})) / 2\nkijun  = (ta.highest(high, {params[1]}) + ta.lowest(low, {params[1]})) / 2\nleadA  = ((tenkan + kijun) / 2)[{params[1]}]\nlongCondition = close > leadA\nshortCondition = close < leadA"

    full_logic = f"{logic}\n"
    if invert:
        full_logic += "// INVERSE MODE\ntemp = longCondition\nlongCondition := shortCondition\nshortCondition := temp\n"

    return f"""//@version=5
strategy("Lux AI: {name} {'(Inv)' if invert else ''}", overlay=true, initial_capital=10000)
{full_logic}
if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)
"""


# ==========================================
# 5. BACKTEST ENGINE
# ==========================================

def compute_core_series(df, signals, capital, invert=False):
    n = len(df)
    sig = ensure_1d_len(signals, n, df.index, 0.0)
    if invert: sig = -sig
    
    pos_arr = np.roll(sig, 1); pos_arr[0] = 0.0
    close_pct = df["Close"].pct_change().fillna(0).to_numpy()
    ret_arr = pos_arr * close_pct
    equity_arr = capital * np.cumprod(1 + ret_arr)
    trade_diff = np.abs(np.diff(pos_arr, prepend=0))
    
    return ret_arr, equity_arr, trade_diff, pos_arr

def evaluate_window(ret_slice, equity_slice, trade_diff_slice, capital_start):
    if len(ret_slice) < 2: return 0, 0, 0, 0, 0
    final_eq = equity_slice[-1]
    pnl_pct = (final_eq - equity_slice[0]) / equity_slice[0] * 100
    
    wins = np.sum(ret_slice[ret_slice > 0])
    losses = np.abs(np.sum(ret_slice[ret_slice < 0]))
    pf = wins / losses if losses > 0 else (99.99 if wins > 0 else 0)
    
    peak = np.maximum.accumulate(equity_slice)
    dd = (equity_slice - peak) / peak
    max_dd = np.abs(np.min(dd)) * 100
    txns = int(np.sum(trade_diff_slice) / 2)
    
    return final_eq, txns, pf, pnl_pct, max_dd


# ==========================================
# 6. MAIN ENDPOINT
# ==========================================

@app.post("/generate-strategy")
async def optimize(req: OptimizationRequest):
    print(f"--> QUANTUM SCAN: {req.symbol}...")
    try:
        ticker = req.symbol.replace("Gold", "GC=F").replace("Silver", "SI=F")
        try:
            raw = yf.download(ticker, period="730d", interval=req.timeframe, progress=False, auto_adjust=True)
        except Exception:
            time.sleep(1)
            raw = yf.download(ticker, period="60d", interval=req.timeframe, progress=False, auto_adjust=True)

        if raw is None or raw.empty: raise HTTPException(status_code=404, detail="Data fetch failed.")
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)

        n_rows = len(raw.index)
        clean_data = {}
        for target in ["Open", "High", "Low", "Close", "Volume"]:
            matches = [c for c in raw.columns if target.lower() in str(c).lower()]
            if not matches:
                if target == "Volume": clean_data[target] = np.zeros(n_rows, dtype=float); continue
                raise ValueError(f"Missing {target}")
            val = np.asarray(raw[matches[0]].values)
            if val.ndim > 1: val = val[:, 0]
            if len(val) > n_rows: val = val[:n_rows]
            elif len(val) < n_rows: val = np.pad(val, (0, n_rows - len(val)), constant_values=np.nan)
            clean_data[target] = val

        df = pd.DataFrame(clean_data, index=raw.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty or len(df) < 300: raise HTTPException(status_code=500, detail="Not enough clean data.")

        n = len(df)
        bars_per_day = 24 if ("-" in ticker or "=" in ticker) else 7
        strategies = []

        def add_strategy(stype, args, sig, tag):
            sig = ensure_1d_len(sig, n, df.index, 0.0)
            assert_sig(f"{stype} {args}", sig, n)
            strategies.append({"type": stype, "args": args, "sig": sig, "tag": tag})

        add_strategy("Engulfing", (), strat_engulfing(df), "Price Action")
        for p in [10, 20]: add_strategy("OBV Trend", (p,), strat_obv(df, p), "Volume")
        for p, l in itertools.product([14], [0.05]): add_strategy("CMF", (p, l), strat_cmf(df, p, l), "Volume")
        for p, t in itertools.product([20], [2.0]): add_strategy("Z-Score", (p, t), strat_zscore(df, p, t), "Statistical")
        for f, s in itertools.product([9, 21], [21, 50]): 
            if f < s: add_strategy("TEMA Cross", (f, s), strat_tema(df, f, s), "Scalp")
        for f, s in itertools.product(range(10, 50, 10), range(50, 150, 25)): 
            if f < s: add_strategy("EMA Cross", (f, s), strat_ema(df, f, s), "Trend")
        for p, l in itertools.product([7, 14], [25, 30]): add_strategy("RSI", (p, l), strat_rsi(df, p, l), "Reversion")
        for p, m in itertools.product([10, 14], [2, 3]): add_strategy("SuperTrend", (p, m), calc_supertrend(df, p, m), "Trend")
        for t, k, s in [(9, 26, 52)]: add_strategy("Ichimoku", (t, k, s), calc_ichimoku(df, t, k, s), "Trend")
        for w, d, adx in itertools.product([20], [2.0], [25]): add_strategy("BB+ADX", (w, d, adx), strat_bb_adx(df, w, d, adx), "Hybrid")
        for l, b, k in [(20, 2.0, 1.5)]: add_strategy("TTM Squeeze", (l, b, k), strat_sqz(df, l, b, k), "Volatility")

        valid_candidates = []
        fallback_candidates = []
        
        for s in strategies:
            for invert in [False, True]:
                ret_arr, equity_arr, trade_diff, _ = compute_core_series(df, s["sig"], req.initial_capital, invert)
                for days in sorted(req.window_days_candidates):
                    window_len = int(days * bars_per_day)
                    if window_len > n: break
                    if window_len < req.min_window_days * bars_per_day: continue
                    
                    start_idx = n - window_len
                    r_slice, e_slice, t_slice = ret_arr[start_idx:], equity_arr[start_idx:], trade_diff[start_idx:]
                    final, txns, pf, pnl, max_dd = evaluate_window(r_slice, e_slice, t_slice, req.initial_capital)
                    
                    res = {
                        "name": f"{s['type']} {s['args']} {'(Inv)' if invert else ''}".strip(),
                        "days": days, "txns": txns, "pf": pf, "pnl": pnl, "dd": max_dd, "final": final,
                        "pine": get_pine_script(s["type"], s["args"], invert),
                        "tag": s["tag"], "curve_slice": e_slice, "start_idx": start_idx
                    }

                    # Constraints Check with MAX TRADES included
                    if txns >= req.min_trades and txns <= req.max_trades and pf >= req.min_profit_factor and max_dd <= req.max_drawdown:
                        valid_candidates.append(res)
                        break
                    
                    if txns > 5:
                        res["score"] = pf / (np.log10((txns / max(req.min_trades, 1)) + 0.001) + 2.0)
                        fallback_candidates.append(res)

        winner, note = None, ""
        if valid_candidates:
            valid_candidates.sort(key=lambda x: (x["days"], -x["pf"], -x["pnl"]))
            winner, note = valid_candidates[0], f"Constraint Met in {valid_candidates[0]['days']} Days"
        else:
            if not fallback_candidates: raise HTTPException(status_code=500, detail="No viable strategies found.")
            fallback_candidates.sort(key=lambda x: x["score"], reverse=True)
            winner, note = fallback_candidates[0], f"Best Fallback ({fallback_candidates[0]['days']} Days)"

        subset_vals = winner["curve_slice"]
        subset_idx = df.index[winner["start_idx"]:]
        downsample = 5 if len(subset_vals) > 1000 else 1
        
        return {
            "strategy_name": winner["name"],
            "window_days": winner["days"],
            "metrics": {
                "total_pnl": winner["final"] - subset_vals[0],
                "pnl_percent": winner["pnl"],
                "total_txns": winner["txns"],
                "profit_factor": round(winner["pf"], 2),
                "pine_code": winner["pine"],
                "max_dd": round(winner["dd"], 2),
                "tag": winner["tag"],
                "duration": f"{winner['days']} Days",
                "note": note,
            },
            "chart_data": [{"time": int(t.timestamp()), "value": float(v)} for t, v in zip(subset_idx[::downsample], subset_vals[::downsample])]
        }

    except HTTPException: raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"use client";

import React, { useState, useRef, useEffect, useMemo } from "react";
import { supabase } from "./supabase"; // Ensure this path is correct
import { useRouter } from "next/navigation";
import { LiveTicker } from "./components/LiveTicker"; // Keep your existing components
import { toast } from "sonner";
import { InteractiveChart } from "./components/InteractiveChart"; // Keep your existing components
import { LegalModal } from "./components/LegalModal"; // Keep your existing components

// --- INTERNAL TERMINAL LOG COMPONENT (Fixed Scrolling) ---
const LOG_MESSAGES = [
  "Initializing neural network...",
  "Connecting to Binance Websocket...",
  "Fetching OHLCV data...",
  "Calculating volatility indices...",
  "Running Monte Carlo simulations (n=10,000)...",
  "Optimizing entry/exit vectors...",
  "Detecting support/resistance levels...",
  "Validating risk parameters...",
  "Finalizing strategy metrics...",
  "Generating Pine Script code..."
];

const TerminalLog = ({ active }: { active: boolean }) => {
  const [logs, setLogs] = useState<string[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!active) return;

    setLogs([]); // Reset logs when activation starts
    let i = 0;

    const interval = setInterval(() => {
      if (i < LOG_MESSAGES.length) {
        const msg = LOG_MESSAGES[i];
        setLogs(prev => [...prev, `> ${msg}`]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 600); 

    return () => clearInterval(interval);
  }, [active]);

  // FIX: This scrolls ONLY the container, not the whole window
  useEffect(() => {
    if (containerRef.current) {
        containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  if (!active) return null;

  return (
    <div 
        ref={containerRef}
        className="font-mono text-xs text-emerald-500/90 bg-black/80 p-6 rounded-xl border border-emerald-500/20 h-64 overflow-y-auto mb-6 backdrop-blur-md shadow-[inset_0_0_20px_rgba(16,185,129,0.1)] scroll-smooth"
    >
      <div className="flex flex-col gap-2">
        {logs.map((log, idx) => (
          <div key={idx} className="animate-in fade-in slide-in-from-left-2 duration-300">
            {log} <span className="text-emerald-800 ml-2">[OK]</span>
          </div>
        ))}
        {/* Blinking cursor */}
        {logs.length < LOG_MESSAGES.length && (
             <div className="animate-pulse text-emerald-400">_</div>
        )}
      </div>
    </div>
  );
};

// --- CONFIGURATION ---

const EXTENDED_ASSETS = [
  {
    category: "Crypto",
    items: [
      { name: "Bitcoin", ticker: "BTC-USD" },
      { name: "Ethereum", ticker: "ETH-USD" },
      { name: "Solana", ticker: "SOL-USD" },
      { name: "Ripple", ticker: "XRP-USD" },
      { name: "Dogecoin", ticker: "DOGE-USD" },
      { name: "Cardano", ticker: "ADA-USD" },
      { name: "Polkadot", ticker: "DOT-USD" },
      { name: "Chainlink", ticker: "LINK-USD" },
      { name: "Litecoin", ticker: "LTC-USD" },
    ],
  },
  {
    category: "Forex",
    items: [
      { name: "EUR/USD", ticker: "EURUSD=X" },
      { name: "GBP/USD", ticker: "GBPUSD=X" },
      { name: "USD/JPY", ticker: "USDJPY=X" },
      { name: "USD/CAD", ticker: "CAD=X" },
      { name: "AUD/USD", ticker: "AUDUSD=X" },
      { name: "USD/CHF", ticker: "CHF=X" },
      { name: "NZD/USD", ticker: "NZDUSD=X" },
    ],
  },
  {
    category: "Metals & Commodities",
    items: [
      { name: "Gold", ticker: "GC=F" },
      { name: "Silver", ticker: "SI=F" },
      { name: "Crude Oil", ticker: "CL=F" },
      { name: "Copper", ticker: "HG=F" },
    ],
  },
  {
    category: "Indices",
    items: [
      { name: "S&P 500", ticker: "ES=F" },
      { name: "Nasdaq 100", ticker: "NQ=F" },
      { name: "Dow Jones", ticker: "YM=F" },
      { name: "Russell 2000", ticker: "RTY=F" },
    ],
  },
  {
    category: "Tech Stocks",
    items: [
      { name: "Nvidia", ticker: "NVDA" },
      { name: "Tesla", ticker: "TSLA" },
      { name: "Apple", ticker: "AAPL" },
      { name: "Microsoft", ticker: "MSFT" },
      { name: "AMD", ticker: "AMD" },
      { name: "Amazon", ticker: "AMZN" },
      { name: "Google", ticker: "GOOGL" },
      { name: "Meta", ticker: "META" },
    ],
  },
];

// --- INTERFACES ---

interface OptimizationRequest {
  symbol: string;
  initial_capital: number;
  min_trades: number;
  max_trades: number;
  min_profit_factor: number;
  max_drawdown: number;
  timeframe: string;
  window_mode: string;
  window_days_candidates?: number[];
}

interface StrategyResponse {
  strategy_name: string;
  window_days: number;
  metrics: {
    total_pnl: number;
    pnl_percent: number;
    total_txns: number;
    profit_factor: number;
    max_dd: number;
    pine_code: string;
    tag: string;
    duration: string;
    note: string;
    start_date: string;
    end_date: string;
  };
  chart_data: Array<{ time: number; value: number }>;
  generatedId?: string;
}

type AssetStatsResponse = { price: number; change: number; volatility: number; stale?: boolean };

// --- ICONS ---
const Icons = {
  Alert: () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  ),
  Code: () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
  ),
  Download: () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  ),
  Save: () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
    </svg>
  ),
  Search: () => (
    <svg className="w-3 h-3 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  ),
  Refresh: () => (
    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  ),
  Calendar: () => (
    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  ),
  Zap: () => (
    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  )
};

const Skeleton = ({ className }: { className: string }) => <div className={`animate-pulse bg-zinc-800/50 rounded ${className}`} />;

const Badge = ({
  children,
  type = "neutral",
}: {
  children: React.ReactNode;
  type?: "neutral" | "success" | "warning" | "danger" | "info";
}) => {
  const styles = {
    neutral: "bg-zinc-800 text-zinc-400 border-zinc-700",
    success: "bg-emerald-950/30 text-emerald-400 border-emerald-900/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]",
    warning: "bg-amber-950/30 text-amber-400 border-amber-900/50",
    danger: "bg-rose-950/30 text-rose-400 border-rose-900/50",
    info: "bg-blue-950/30 text-blue-400 border-blue-900/50",
  };
  return <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border ${styles[type]}`}>{children}</span>;
};

// --- UTIL: BACKEND URL + SAFE FETCH ---

const backendBase = () => (process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function safeJson(res: Response) {
  const text = await res.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { detail: text || "Invalid JSON response" };
  }
}

function genId() {
  return Math.random().toString(36).slice(2, 11).toUpperCase();
}

// --- SYMBOL ACTIVATION (BEST EFFORT) ---
async function activateSymbol(ticker: string) {
  const base = backendBase();
  try {
    // if your backend supports /activate-symbol?symbol=...
    const r = await fetch(`${base}/activate-symbol?symbol=${encodeURIComponent(ticker)}`, { method: "POST", cache: "no-store" });
    void r;
  } catch {
    // silent
  }
}

export default function Terminal() {
  const [loading, setLoading] = useState(false);
  const [strategy, setStrategy] = useState<StrategyResponse | null>(null);
  const [error, setError] = useState("");
  const [showCode, setShowCode] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");
  const [user, setUser] = useState<any>(null);
  const [savedStrategies, setSavedStrategies] = useState<any[]>([]);
  const [assetStats, setAssetStats] = useState<AssetStatsResponse>({ price: 0, change: 0, volatility: 0 });

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<"idle" | "queued" | "running" | "done" | "error">("idle");

  const [modalOpen, setModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState({ title: "", text: "" });

  const abortControllerRef = useRef<AbortController | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const router = useRouter();

  // Inputs
  const [symbol, setSymbol] = useState("BTC-USD");
  const [displayName, setDisplayName] = useState("Bitcoin");
  const [capital, setCapital] = useState(50000);
  const [minTrades, setMinTrades] = useState(25);
  const [maxTrades, setMaxTrades] = useState(150);
  const [targetPF, setTargetPF] = useState(1.5);
  const [riskMode, setRiskMode] = useState<"Funded" | "Live">("Funded");
  const [windowMode, setWindowMode] = useState("auto_shortest");

  const [isAssetOpen, setIsAssetOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const dashboardRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const filteredAssets = useMemo(() => {
    if (!searchQuery) return EXTENDED_ASSETS;
    const lowerQ = searchQuery.toLowerCase();
    return EXTENDED_ASSETS
      .map((cat) => ({
        ...cat,
        items: cat.items.filter((item) => item.name.toLowerCase().includes(lowerQ) || item.ticker.toLowerCase().includes(lowerQ)),
      }))
      .filter((cat) => cat.items.length > 0);
  }, [searchQuery]);

  const applyPreset = (type: "prop" | "agg" | "high") => {
    if (type === "prop") {
      setMinTrades(30);
      setTargetPF(1.6);
      setRiskMode("Funded");
      toast.info("Applied: Prop Firm Safe");
    }
    if (type === "agg") {
      setMinTrades(15);
      setTargetPF(1.2);
      setRiskMode("Live");
      toast.info("Applied: Aggressive Growth");
    }
    if (type === "high") {
      setMinTrades(60);
      setTargetPF(1.3);
      setWindowMode("multi_recent");
      toast.info("Applied: High Sample Size");
    }
  };

  // AUTH
  useEffect(() => {
    const init = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      setUser(user);
      if (user) fetchSavedStrategies(user.id);
    };
    init();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      if (session?.user) fetchSavedStrategies(session.user.id);
    });

    return () => subscription.unsubscribe();
  }, []);

  const fetchSavedStrategies = async (uid: string) => {
    const { data } = await supabase
      .from("strategies")
      .select("name, pf, trades, symbol")
      .eq("user_id", uid)
      .order("created_at", { ascending: false })
      .limit(3);

    if (data) setSavedStrategies(data);
  };

  // BEST-EFFORT ACTIVATE ON SYMBOL CHANGE (DEBOUNCED)
  useEffect(() => {
    let cancelled = false;
    const t = window.setTimeout(async () => {
      if (cancelled) return;
      await activateSymbol(symbol);
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [symbol]);

  // ASSET STATS (DEBOUNCED)
  useEffect(() => {
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch(`${backendBase()}/asset-stats?symbol=${encodeURIComponent(symbol)}`, { cache: "no-store" });
        if (!res.ok) return;
        const json = (await safeJson(res)) as AssetStatsResponse;
        if (!cancelled) setAssetStats(json);
      } catch {
        // silent
      }
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [symbol]);

  // HEALTH
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await fetch(backendBase(), { method: "HEAD", cache: "no-store" });
        setBackendStatus("online");
      } catch {
        setBackendStatus("offline");
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // CLICK OUTSIDE DROPDOWN
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const el = dropdownRef.current;
      if (!el) return;
      if (!el.contains(event.target as Node)) setIsAssetOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogin = async () => {
    const email = prompt("Enter email for Magic Link login:");
    if (!email) return;
    const toastId = toast.loading("Sending magic link...");
    const { error } = await supabase.auth.signInWithOtp({ email });
    if (error) toast.error(error.message, { id: toastId });
    else toast.success("Check your email! Magic link sent.", { id: toastId });
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    toast.info("Logged out successfully");
  };

  const saveStrategy = async () => {
    if (!user) {
      toast.error("Please login to save strategies!");
      handleLogin();
      return;
    }
    if (!strategy) return;

    const toastId = toast.loading("Saving to dashboard...");
    const { error } = await supabase.from("strategies").insert({
      user_id: user.id,
      symbol: symbol,
      name: strategy.strategy_name,
      entry_price: strategy.chart_data[strategy.chart_data.length - 1]?.value || 100,
      pf: strategy.metrics.profit_factor,
      win_rate: 0,
      trades: strategy.metrics.total_txns,
      duration: strategy.metrics.duration,
      pine_code: strategy.metrics.pine_code,
    });

    if (error) toast.error("Error saving: " + error.message, { id: toastId });
    else {
      toast.success("Strategy saved!", { id: toastId });
      fetchSavedStrategies(user.id);
    }
  };

  const handleCopyCode = async () => {
    if (!strategy?.metrics?.pine_code) {
      toast.error("No code available");
      return;
    }
    try {
      await navigator.clipboard.writeText(strategy.metrics.pine_code);
      toast.success("Copied to clipboard");
    } catch {
      toast.error("Failed to copy");
    }
  };

  const handleReset = () => {
    if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
    pollTimerRef.current = null;

    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = null;

    setJobId(null);
    setJobStatus("idle");
    setStrategy(null);
    setError("");
    setShowCode(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const openModal = (type: "docs" | "terms" | "privacy") => {
    const contentMap = {
      docs: {
        title: "Documentation",
        text: "Lux Quant AI scans historical regimes and returns a best-fit rule set for the chosen constraints. Results are backtest simulations, not financial advice.",
      },
      terms: { title: "Terms", text: "No financial advice. Use at your own risk. Past performance does not guarantee future results." },
      privacy: { title: "Privacy", text: "We do not sell your data." },
    };
    setModalContent(contentMap[type]);
    setModalOpen(true);
  };

  const scrollToDashboard = () => dashboardRef.current?.scrollIntoView({ behavior: "smooth" });

  // --- UPDATED OPTIMIZATION LOGIC ---

  // NOTE: Added optional 'overrides' to support "Find Similar" functionality
  const runOptimization = async (overrides?: Partial<OptimizationRequest>) => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
    pollTimerRef.current = null;

    setLoading(true);
    setStrategy(null);
    setError("");
    setShowCode(false);

    setJobStatus("queued");
    setJobId(null);

    await activateSymbol(symbol);

    const maxDrawdown = riskMode === "Funded" ? 10.0 : 30.0;

    // Merge current state with any overrides
    const payload: OptimizationRequest = {
      symbol,
      initial_capital: capital,
      min_trades: minTrades,
      max_trades: maxTrades,
      min_profit_factor: targetPF,
      max_drawdown: maxDrawdown,
      timeframe: "1h",
      window_mode: windowMode,
      window_days_candidates: [30, 45, 60, 90, 120, 180, 365],
      ...overrides // Overrides apply last
    };

    const BASE = backendBase();
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

    try {
      const res = await fetch(`${BASE}/generate-strategy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const data = await safeJson(res);

      if (!res.ok) {
        const retryAfter = res.headers.get("Retry-After");
        const msg = (data as any)?.detail || "Optimization failed";
        throw new Error(retryAfter ? `${msg} (retry after ${retryAfter}s)` : msg);
      }

      if ((data as any)?.cached && (data as any)?.result) {
        const result = (data as any).result as StrategyResponse;
        setStrategy({ ...result, generatedId: genId() });
        setJobId(null);
        setJobStatus("done");
        return;
      }

      const jid = (data as any)?.job_id as string | undefined;
      if (!jid) throw new Error("No job_id returned from backend.");

      setJobId(jid);
      setJobStatus(((data as any)?.status as any) === "running" ? "running" : "queued");

      const pollMaxMs = 60_000;
      const started = Date.now();
      let waitMs = 700;

      while (Date.now() - started < pollMaxMs) {
        if (controller.signal.aborted) return;

        const jr = await fetch(`${BASE}/job/${encodeURIComponent(jid)}`, { cache: "no-store", signal: controller.signal });
        const jd = await safeJson(jr);

        if (!jr.ok) {
          const msg = (jd as any)?.detail || "Job polling failed";
          throw new Error(msg);
        }

        if ((jd as any).status === "done" && (jd as any).result) {
          setStrategy({ ...((jd as any).result as StrategyResponse), generatedId: genId() });
          setJobId(null);
          setJobStatus("done");
          return;
        }

        if ((jd as any).status === "error") {
          throw new Error((jd as any).error || "Optimization job failed");
        }

        setJobStatus((jd as any).status === "running" ? "running" : "queued");

        await sleep(waitMs);
        waitMs = Math.min(2500, Math.floor(waitMs * 1.25));
      }

      throw new Error("Timed out waiting for optimization result. Try again.");
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setJobStatus("error");
      setJobId(null);
      setError(e?.message || "Server Offline. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // --- NEW: FIND SIMILAR HANDLER ---
  const handleFindSimilar = () => {
    // We slightly tweak the settings to force the backend to find a DIFFERENT optimum
    // 1. Randomize trade count by +/- 10-20%
    // 2. Tweak Profit Factor slightly
    
    const variance = Math.floor(Math.random() * 10) + 5; // Random number between 5 and 15
    const direction = Math.random() > 0.5 ? 1 : -1;
    
    // Calculate new constraints
    const newMinTrades = Math.max(5, minTrades + (variance * direction));
    const newMaxTrades = maxTrades + (variance * direction);
    const newTargetPF = Math.max(1.1, targetPF + (Math.random() * 0.2 - 0.1)); // Tweak PF by +/- 0.1

    toast.info(`Finding variations... (Trades: ${newMinTrades}-${newMaxTrades}, PF: ${newTargetPF.toFixed(1)})`);

    // Run optimization with these overrides
    runOptimization({
        min_trades: newMinTrades,
        max_trades: newMaxTrades,
        min_profit_factor: newTargetPF
    });
  };

  const downloadCSV = () => {
    if (!strategy) return;
    const headers = "Time,Equity\n";
    const rows = strategy.chart_data.map((d) => `${new Date(d.time * 1000).toISOString()},${d.value}`).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${symbol}_strategy_data.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getConfidenceLevel = (metrics: any) => {
    if (metrics.total_txns < minTrades + 5) return { label: "LOW DENSITY", type: "warning" as const };
    if (metrics.profit_factor < targetPF + 0.1) return { label: "MARGINAL", type: "warning" as const };
    if (metrics.max_dd > (riskMode === "Funded" ? 8 : 25)) return { label: "HIGH RISK", type: "danger" as const };
    return { label: "HIGH CONFIDENCE", type: "success" as const };
  };

  const jobPill = () => {
    if (jobStatus === "idle") return null;
    if (jobStatus === "queued") return <Badge type="info">QUEUED</Badge>;
    if (jobStatus === "running") return <Badge type="warning">RUNNING</Badge>;
    if (jobStatus === "done") return <Badge type="success">DONE</Badge>;
    return <Badge type="danger">ERROR</Badge>;
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500/30 relative overflow-x-hidden">
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Background Grids and Glows */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_800px_at_50%_200px,#00000000,transparent)]" />
        <div className="absolute top-[30%] left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-emerald-500/5 blur-[120px] rounded-full pointer-events-none" />
      </div>

      {/* NAVBAR */}
      <nav className="fixed w-full z-50 bg-zinc-950/80 backdrop-blur-md border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center">
            <div className="text-xl font-bold tracking-tighter flex items-center gap-2 text-white mr-6">
              <div
                className={`w-2 h-2 rounded-full animate-pulse shadow-[0_0_10px_currentColor] ${
                  backendStatus === "online" ? "bg-emerald-500 text-emerald-500" : "bg-red-500 text-red-500"
                }`}
              />
              LUX QUANT <span className="text-emerald-500">FACTORY</span>
            </div>
            <LiveTicker />
          </div>
          <div className="flex items-center gap-4">
            {user ? (
              <>
                <button onClick={() => router.push("/dashboard")} className="text-xs text-zinc-400 hover:text-white transition-colors">
                  DASHBOARD
                </button>
                <button onClick={handleLogout} className="text-xs text-zinc-400 hover:text-white transition-colors">
                  LOGOUT
                </button>
              </>
            ) : (
              <button onClick={handleLogin} className="text-xs text-zinc-400 hover:text-white transition-colors">
                LOGIN
              </button>
            )}
            <button
              onClick={scrollToDashboard}
              className="bg-zinc-100 text-zinc-950 px-5 py-2 rounded-full text-xs font-bold hover:bg-zinc-200 transition-colors border border-transparent hover:border-emerald-500 shadow-[0_0_15px_rgba(255,255,255,0.1)]"
            >
              LAUNCH TERMINAL
            </button>
          </div>
        </div>
      </nav>

      {/* HERO */}
      <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-6 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-emerald-900/20 via-zinc-950 to-zinc-950 z-0" />
        <div className="max-w-5xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-[10px] font-bold tracking-[0.2em] uppercase animate-in fade-in slide-in-from-bottom-4 duration-700 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
            <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full shadow-[0_0_5px_currentColor]" /> v4.4 ECOSYSTEM LIVE
          </div>
          <h1 className="text-5xl md:text-8xl font-bold tracking-tighter mb-10 pb-4 leading-tight bg-gradient-to-b from-white via-zinc-200 to-zinc-500 bg-clip-text text-transparent animate-in fade-in zoom-in-95 duration-1000 drop-shadow-2xl">
            Mathematically Proven <br /> Trading Alpha.
          </h1>
          <div className="flex flex-col md:flex-row gap-4 justify-center items-center animate-in fade-in slide-in-from-bottom-4 delay-300 duration-1000">
            <button
              onClick={scrollToDashboard}
              className="group relative bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-4 rounded-xl text-sm font-bold transition-all w-full md:w-auto overflow-hidden shadow-[0_0_20px_rgba(16,185,129,0.4)] hover:shadow-[0_0_30px_rgba(16,185,129,0.6)]"
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
              <span className="relative flex items-center gap-2 justify-center">START GENERATING ▼</span>
            </button>
            <p className="text-xs text-zinc-600 mt-4 md:mt-0">
              *Processing power required.
              <br />
              Results may take 3-20 seconds.
            </p>
          </div>
        </div>
      </section>

      {/* DASHBOARD */}
      <section ref={dashboardRef} className="py-24 px-4 md:px-8 relative z-10">
        <div className="max-w-7xl mx-auto relative z-10">
          {/* HEADER */}
          <div className="flex flex-col md:flex-row items-end justify-between border-b border-white/5 pb-6 mb-10 gap-4">
            <div>
              <h2 className="text-3xl font-bold text-white tracking-tight drop-shadow-lg">Strategy Terminal</h2>
              <p className="text-sm text-zinc-500 mt-1">Configure parameters. The engine returns the best fit for your constraints.</p>
            </div>
            <div className="flex gap-3">
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-zinc-500 font-bold uppercase mb-1">System Status</span>
                <Badge type={backendStatus === "online" ? "success" : "danger"}>{backendStatus === "online" ? "ONLINE" : "OFFLINE"}</Badge>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-zinc-500 font-bold uppercase mb-1">Data Feed</span>
                <Badge type="info">CACHE-ONLY</Badge>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-zinc-500 font-bold uppercase mb-1">Job</span>
                {jobPill() || <Badge type="neutral">IDLE</Badge>}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* SIDEBAR */}
            <div className="lg:col-span-3 space-y-6 sticky top-24">
              {/* ASSET SELECTOR */}
              <div className="relative z-50 bg-zinc-900/50 p-6 rounded-2xl border border-white/5 backdrop-blur-md shadow-[0_8px_30px_rgba(0,0,0,0.5)] hover:border-emerald-500/20 transition-all duration-500">
                <div className="space-y-4">
                  <div className="relative" ref={dropdownRef}>
                    <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Asset Class</label>
                    <button
                      onClick={() => setIsAssetOpen(!isAssetOpen)}
                      className="w-full mt-2 bg-zinc-950 border border-white/10 text-white py-3 px-4 rounded-lg flex justify-between items-center hover:border-emerald-500/50 hover:shadow-[0_0_15px_rgba(16,185,129,0.1)] transition-all text-sm font-mono group"
                    >
                      <span className="truncate">{displayName}</span>
                      <span className={`text-zinc-600 group-hover:text-emerald-500 transition-transform ${isAssetOpen ? "rotate-180" : ""}`}>▼</span>
                    </button>

                    {isAssetOpen && (
                      <div className="absolute z-[100] w-full mt-2 bg-zinc-950 border border-zinc-700 rounded-xl shadow-2xl max-h-80 overflow-y-auto overflow-x-hidden animate-in fade-in slide-in-from-top-2 duration-200">
                        <div className="sticky top-0 bg-zinc-950 p-2 border-b border-white/10 z-[101]">
                          <div className="relative">
                            <input
                              autoFocus
                              type="text"
                              placeholder="Search assets..."
                              className="w-full bg-zinc-900 text-xs text-white p-2 pl-8 rounded border border-white/10 focus:border-emerald-500/50 outline-none"
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                            />
                            <div className="absolute left-2.5 top-2.5">
                              <Icons.Search />
                            </div>
                          </div>
                        </div>

                        {filteredAssets.length === 0 ? (
                          <div className="p-4 text-center text-xs text-zinc-500">No assets found.</div>
                        ) : (
                          filteredAssets.map((cat, i) => (
                            <div key={i} className="p-2 border-b border-white/5 last:border-0">
                              <div className="text-[9px] text-zinc-500 font-bold px-2 py-1 uppercase">{cat.category}</div>
                              {cat.items.map((item, j) => (
                                <div
                                  key={j}
                                  onClick={() => {
                                    activateSymbol(item.ticker);
                                    setSymbol(item.ticker);
                                    setDisplayName(item.name);
                                    setIsAssetOpen(false);
                                    setSearchQuery("");
                                  }}
                                  className="flex justify-between items-center px-2 py-2 text-xs text-zinc-300 hover:bg-emerald-900/20 hover:text-emerald-400 rounded cursor-pointer font-mono transition-colors group"
                                >
                                  <span>{item.name}</span>
                                  {searchQuery && <span className="text-[9px] text-zinc-600 group-hover:text-emerald-600">{item.ticker}</span>}
                                </div>
                              ))}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>

                  {/* LIVE SNAPSHOT WIDGET */}
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-950/50 p-3 rounded border border-white/5 shadow-inner">
                    <div>
                      <div className="text-zinc-500 text-[10px]">Price</div>
                      <div className="text-white">${Number(assetStats.price || 0).toFixed(2)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-zinc-500 text-[10px]">24h</div>
                      <div className={(assetStats.change || 0) >= 0 ? "text-emerald-400" : "text-rose-400"}>
                        {(assetStats.change || 0) > 0 ? "+" : ""}
                        {Number(assetStats.change || 0).toFixed(2)}%
                      </div>
                    </div>

                    <div className="col-span-2 mt-2">
                      <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                        <span>Volatility Heat</span>
                        <span>
                          {Number(assetStats.volatility || 0).toFixed(1)}
                          {assetStats.stale ? " (stale)" : ""}
                        </span>
                      </div>
                      <div className="w-full bg-zinc-800 h-1 rounded-full overflow-hidden">
                        <div
                          className="bg-orange-500 h-full transition-all duration-1000 shadow-[0_0_10px_orange]"
                          style={{ width: `${Math.min(Number(assetStats.volatility || 0) * 2, 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* QUICK PRESETS & INPUTS */}
              <div className="relative z-10 bg-zinc-900/50 p-6 rounded-2xl border border-white/5 backdrop-blur-md shadow-[0_8px_30px_rgba(0,0,0,0.5)] hover:border-emerald-500/20 transition-all duration-500">
                <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider mb-2 block">Quick Presets</label>
                <div className="grid grid-cols-3 gap-2 mb-6">
                  <button
                    onClick={() => applyPreset("prop")}
                    className="p-2 bg-zinc-950 border border-white/10 rounded hover:border-emerald-500/50 text-[9px] font-bold text-zinc-400 hover:text-emerald-400 transition-all hover:shadow-[0_0_10px_rgba(16,185,129,0.2)]"
                  >
                    PROP SAFE
                  </button>
                  <button
                    onClick={() => applyPreset("agg")}
                    className="p-2 bg-zinc-950 border border-white/10 rounded hover:border-blue-500/50 text-[9px] font-bold text-zinc-400 hover:text-blue-400 transition-all hover:shadow-[0_0_10px_rgba(59,130,246,0.2)]"
                  >
                    AGGRO
                  </button>
                  <button
                    onClick={() => applyPreset("high")}
                    className="p-2 bg-zinc-950 border border-white/10 rounded hover:border-purple-500/50 text-[9px] font-bold text-zinc-400 hover:text-purple-400 transition-all hover:shadow-[0_0_10px_rgba(168,85,247,0.2)]"
                  >
                    SAMPLE+
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Initial Capital ($)</label>
                    <input
                      type="number"
                      value={capital}
                      onChange={(e) => setCapital(Number(e.target.value))}
                      className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-white font-mono text-sm focus:border-emerald-500/50 focus:shadow-[0_0_10px_rgba(16,185,129,0.1)] outline-none transition-all"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Min Trades</label>
                      <input
                        type="number"
                        value={minTrades}
                        onChange={(e) => setMinTrades(Number(e.target.value))}
                        className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-center font-mono text-sm focus:border-blue-500/50 outline-none"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Max Trades</label>
                      <input
                        type="number"
                        value={maxTrades}
                        onChange={(e) => setMaxTrades(Number(e.target.value))}
                        className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-center font-mono text-sm focus:border-blue-500/50 outline-none"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Target PF</label>
                    <input
                      type="number"
                      step="0.1"
                      value={targetPF}
                      onChange={(e) => setTargetPF(Number(e.target.value))}
                      className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-center font-mono text-sm focus:border-emerald-500/50 outline-none"
                    />
                  </div>

                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Risk Mode</label>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <button
                        onClick={() => setRiskMode("Funded")}
                        className={`py-3 rounded-lg text-[10px] font-bold transition-all border ${
                          riskMode === "Funded"
                            ? "bg-emerald-900/20 border-emerald-500 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)]"
                            : "bg-zinc-950 border-white/10 text-zinc-500 hover:border-zinc-700"
                        }`}
                      >
                        FUNDED (10%)
                      </button>
                      <button
                        onClick={() => setRiskMode("Live")}
                        className={`py-3 rounded-lg text-[10px] font-bold transition-all border ${
                          riskMode === "Live" ? "bg-blue-900/20 border-blue-500 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.2)]" : "bg-zinc-950 border-white/10 text-zinc-500 hover:border-zinc-700"
                        }`}
                      >
                        LIVE (30%)
                      </button>
                    </div>
                  </div>

                  <button
                    onClick={() => runOptimization()}
                    disabled={loading || backendStatus !== "online"}
                    className="w-full bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white py-4 rounded-xl text-sm font-bold transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] disabled:opacity-50 disabled:cursor-not-allowed mt-4 relative overflow-hidden group border border-emerald-400/20"
                    title={backendStatus !== "online" ? "Backend offline" : undefined}
                  >
                    {loading ? (
                      <div className="flex items-center justify-center gap-3">
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span className="tracking-wide text-xs">{jobStatus === "queued" ? "QUEUED..." : jobStatus === "running" ? "RUNNING..." : "CRUNCHING..."}</span>
                      </div>
                    ) : (
                      <span className="group-hover:tracking-widest transition-all duration-300">FIND ALPHA</span>
                    )}
                  </button>

                  {jobId && (
                    <div className="text-[10px] text-zinc-500 font-mono mt-2 flex items-center justify-between">
                      <span>Job: {jobId.slice(0, 10)}...</span>
                      <button
                        className="text-zinc-400 hover:text-white underline underline-offset-4"
                        onClick={() => {
                          if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current);
                          pollTimerRef.current = null;
                          if (abortControllerRef.current) abortControllerRef.current.abort();
                          abortControllerRef.current = null;
                          setLoading(false);
                          setJobStatus("idle");
                          setJobId(null);
                          toast.info("Stopped polling. You can restart the search anytime.");
                        }}
                      >
                        stop
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* SAVED STRATEGIES */}
              {user && savedStrategies.length > 0 && (
                <div className="bg-zinc-900/50 p-6 rounded-2xl border border-white/5 backdrop-blur-md shadow-[0_8px_30px_rgba(0,0,0,0.5)]">
                  <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider mb-3 block">Saved Alpha</label>
                  <div className="space-y-2">
                    {savedStrategies.map((s, i) => (
                      <div key={i} className="flex justify-between items-center text-xs p-2 bg-zinc-950/50 rounded border border-white/5 hover:border-emerald-500/30 transition-colors">
                        <div>
                          <span className="text-white font-bold">{s.symbol}</span> <span className="text-zinc-500">{String(s.name).substring(0, 10)}...</span>
                        </div>
                        <div className="font-mono text-emerald-400">PF: {s.pf}</div>
                      </div>
                    ))}
                    <button onClick={() => router.push("/dashboard")} className="w-full text-center text-[10px] text-zinc-500 hover:text-white mt-2">
                      VIEW ALL →
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* MAIN DISPLAY */}
            <div className="lg:col-span-9 space-y-6 relative z-20">
              {error && (
                <div className="bg-rose-950/10 border border-rose-900/30 text-rose-400 p-6 rounded-2xl flex items-center gap-4 animate-in fade-in slide-in-from-top-2 shadow-[0_0_20px_rgba(244,63,94,0.1)]">
                  <Icons.Alert />
                  <div>
                    <p className="text-sm font-bold">Optimization Failed</p>
                    <p className="text-xs opacity-80 mt-1 font-mono">{error}</p>
                  </div>
                </div>
              )}

              {!strategy && !error && !loading && (
                <div className="relative z-20 min-h-[520px] flex flex-col items-center justify-center text-zinc-600 border border-dashed border-white/10 rounded-2xl bg-zinc-900/20 p-8 backdrop-blur-sm">
                  {/* REFACTORED STEPPER FOR ALIGNMENT */}
                  <div className="max-w-md w-full">
                    {/* Step 1 */}
                    <div className="flex gap-4">
                        <div className="flex flex-col items-center">
                            <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold border border-emerald-500/50 shrink-0 z-10 shadow-[0_0_15px_rgba(16,185,129,0.4)] animate-pulse">1</div>
                            <div className="w-0.5 h-full bg-gradient-to-b from-emerald-500/50 to-white/10 -mt-2 pb-2 min-h-[20px]" />
                        </div>
                        <div className="pb-8 pt-1">
                            <div className="text-sm font-bold text-white drop-shadow-md">Select Asset</div>
                            <div className="text-xs text-zinc-500">Choose a symbol from the sidebar.</div>
                        </div>
                    </div>

                    {/* Step 2 */}
                    <div className="flex gap-4">
                        <div className="flex flex-col items-center">
                            <div className="w-8 h-8 rounded-full bg-zinc-800 text-zinc-400 flex items-center justify-center text-xs font-bold border border-white/10 shrink-0 z-10">2</div>
                            <div className="w-0.5 h-full bg-white/10 -mt-2 pb-2 min-h-[20px]" />
                        </div>
                        <div className="pb-8 pt-1">
                            <div className="text-sm font-bold opacity-60">Set Constraints</div>
                            <div className="text-xs text-zinc-500">Define risk tolerance and trade count.</div>
                        </div>
                    </div>

                    {/* Step 3 */}
                    <div className="flex gap-4">
                        <div className="flex flex-col items-center">
                            <div className="w-8 h-8 rounded-full bg-zinc-800 text-zinc-400 flex items-center justify-center text-xs font-bold border border-white/10 shrink-0 z-10">3</div>
                        </div>
                        <div className="pt-1">
                            <div className="text-sm font-bold opacity-60">Run Job</div>
                            <div className="text-xs text-zinc-500">Queued requests run sequentially under load.</div>
                        </div>
                    </div>
                  </div>
                </div>
              )}

              {loading && (
                <div className="space-y-6">
                  <TerminalLog active={loading} />
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">{[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-32 w-full rounded-xl" />)}</div>
                  <Skeleton className="h-[500px] w-full rounded-2xl" />
                </div>
              )}

              {strategy && !loading && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6">
                  <div className="bg-gradient-to-r from-zinc-900 to-zinc-950 border border-white/10 p-6 rounded-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4 shadow-[0_10px_40px_rgba(0,0,0,0.5)]">
                    <div>
                      <h3 className="text-lg font-bold text-white">
                        Strategy Found: <span className="text-emerald-400 drop-shadow-[0_0_5px_rgba(52,211,153,0.8)]">{strategy.strategy_name}</span>
                      </h3>
                      <div className="flex items-center gap-3 text-xs text-zinc-400 mt-1">
                        <div className="flex items-center gap-1">
                          <Icons.Calendar /> {strategy.metrics.start_date} <span className="text-zinc-600">→</span> {strategy.metrics.end_date}
                        </div>
                        <span className="text-zinc-700">|</span>
                        <span className="text-zinc-200 font-mono">{((strategy.metrics.total_txns / (strategy.window_days || 1)) * 7).toFixed(1)} trades/week</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      {/* FIND SIMILAR BUTTON */}
                      <button
                        onClick={handleFindSimilar}
                        disabled={loading || backendStatus !== "online"}
                        className="flex items-center gap-2 text-[10px] bg-emerald-950/50 hover:bg-emerald-900 text-emerald-400 px-3 py-1.5 rounded border border-emerald-900/50 transition-all hover:shadow-[0_0_15px_rgba(16,185,129,0.2)] font-bold group"
                      >
                         <Icons.Zap /> <span className="group-hover:text-emerald-300">FIND SIMILAR</span>
                      </button>

                      <button
                        onClick={handleReset}
                        className="flex items-center gap-2 text-[10px] bg-zinc-950 hover:bg-zinc-800 text-zinc-400 px-3 py-1.5 rounded border border-white/10 transition-colors font-bold"
                      >
                        <Icons.Refresh /> NEW SEARCH
                      </button>
                      <div className="flex flex-col items-end gap-2">
                        <Badge type={getConfidenceLevel(strategy.metrics).type}>{getConfidenceLevel(strategy.metrics).label}</Badge>
                        <span className="text-[10px] text-zinc-600 font-mono">ID: {strategy.generatedId || "GEN-001"}</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-md shadow-lg hover:border-emerald-500/20 transition-all">
                      <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Net Profit</p>
                      <p className={`text-2xl font-mono font-bold ${strategy.metrics.total_pnl >= 0 ? "text-emerald-400" : "text-red-400"} drop-shadow-md`}>
                        {strategy.metrics.total_pnl >= 0 ? "+" : ""}
                        ${strategy.metrics.total_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </p>
                      <p className="text-xs text-zinc-500 mt-1 font-mono">{strategy.metrics.pnl_percent.toFixed(2)}% Return</p>
                    </div>

                    <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-md shadow-lg hover:border-emerald-500/20 transition-all">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Profit Factor</p>
                          <p className={`text-2xl font-mono font-bold ${strategy.metrics.profit_factor >= targetPF ? "text-emerald-400" : "text-amber-500"} drop-shadow-md`}>
                            {strategy.metrics.profit_factor}
                          </p>
                        </div>
                        <div className="text-[10px] bg-zinc-950 px-2 py-1 rounded text-zinc-400 border border-white/5">{strategy.metrics.note}</div>
                      </div>
                    </div>

                    <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-md shadow-lg hover:border-emerald-500/20 transition-all">
                      <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Drawdown</p>
                      <p className={`text-2xl font-mono font-bold ${strategy.metrics.max_dd > (riskMode === "Funded" ? 9 : 25) ? "text-rose-400" : "text-zinc-200"}`}>
                        {strategy.metrics.max_dd}%
                      </p>
                      <p className="text-xs text-zinc-500 mt-1">Peak-to-Valley Risk</p>
                    </div>

                    <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-md shadow-lg hover:border-emerald-500/20 transition-all">
                      <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Volume</p>
                      <div className="flex items-baseline gap-2">
                        <p className="text-2xl font-mono font-bold text-blue-400">{strategy.metrics.total_txns}</p>
                        <span className="text-xs text-zinc-500">Trades</span>
                      </div>
                      <p className="text-xs text-zinc-500 mt-1 font-mono">{strategy.metrics.duration}</p>
                    </div>
                  </div>

                  <div className="bg-zinc-900/50 p-6 rounded-2xl border border-white/5 flex flex-col shadow-2xl backdrop-blur-md">
                    <div className="flex justify-between items-center mb-6">
                      <div>
                        <h3 className="font-bold text-white text-lg flex items-center gap-2">
                          Equity Curve <Badge type="info">{symbol}</Badge>
                        </h3>
                        <p className="text-xs text-zinc-500 font-mono mt-1">Simulated performance. Static view.</p>
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={saveStrategy}
                          className="flex items-center gap-2 text-[10px] bg-emerald-950/30 text-emerald-400 hover:bg-emerald-900/50 px-3 py-1.5 rounded border border-emerald-900/50 transition-colors font-bold shadow-[0_0_10px_rgba(16,185,129,0.1)]"
                        >
                          <Icons.Save /> SAVE
                        </button>
                        <button
                          onClick={downloadCSV}
                          className="flex items-center gap-2 text-[10px] bg-zinc-950 hover:bg-zinc-800 text-zinc-300 px-3 py-1.5 rounded border border-white/10 transition-colors font-bold"
                        >
                          <Icons.Download /> CSV
                        </button>
                        <button
                          onClick={() => setShowCode(!showCode)}
                          className="flex items-center gap-2 text-[10px] bg-emerald-950/30 text-emerald-500 hover:bg-emerald-950/50 px-3 py-1.5 rounded border border-emerald-900/50 transition-colors font-bold"
                        >
                          <Icons.Code /> {showCode ? "HIDE PINE" : "VIEW PINE"}
                        </button>
                      </div>
                    </div>

                    {showCode ? (
                      <div className="bg-zinc-950 border border-white/5 rounded-xl overflow-hidden flex flex-col h-[420px] shadow-inner">
                        <div className="bg-zinc-900 px-4 py-3 flex justify-between items-center border-b border-white/5 shrink-0">
                          <span className="text-[10px] text-zinc-400 font-bold uppercase tracking-wider">TradingView Pine Script v5</span>
                          <button onClick={handleCopyCode} className="text-[10px] bg-zinc-100 text-black px-3 py-1 rounded hover:bg-zinc-300 font-bold transition-colors">
                            COPY CODE
                          </button>
                        </div>
                        <textarea
                          readOnly
                          value={strategy.metrics.pine_code}
                          className="w-full flex-grow bg-zinc-950 text-xs font-mono text-emerald-400/90 p-4 outline-none resize-none overflow-auto whitespace-pre"
                        />
                      </div>
                    ) : (
                      <div className="w-full h-[420px] border border-white/5 rounded-xl overflow-hidden bg-zinc-950/50 shadow-inner">
                        {strategy.chart_data?.length >= 2 ? (
                          <InteractiveChart data={strategy.chart_data} />
                        ) : (
                          <div className="h-full flex items-center justify-center text-zinc-500 text-xs font-mono">No chart data available.</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-12 border-t border-white/5 bg-zinc-950 relative z-10">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-xs text-zinc-600">&copy; {new Date().getFullYear()} Lux Quant AI. Not financial advice. Data provided for educational purposes.</div>
          <div className="flex gap-6 text-xs text-zinc-500 font-bold uppercase tracking-wider">
            <button onClick={() => openModal("docs")} className="hover:text-white transition-colors">
              Documentation
            </button>
            <button onClick={() => openModal("terms")} className="hover:text-white transition-colors">
              Terms
            </button>
            <button onClick={() => openModal("privacy")} className="hover:text-white transition-colors">
              Privacy
            </button>
          </div>
        </div>
      </footer>

      {/* MODAL (ONLY ONCE) */}
      <LegalModal isOpen={modalOpen} onClose={() => setModalOpen(false)} title={modalContent.title} content={modalContent.text} />
    </main>
  );
}
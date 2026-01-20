"use client";

import React, { useState, useRef, useEffect } from 'react';
import { StrategyChart } from './components/StrategyChart'; 
import { supabase } from './supabase'; 
import { useRouter } from 'next/navigation';
import { LiveTicker } from './components/LiveTicker';
import { toast } from 'sonner';

// --- 1. DEFINITIONS & INTERFACES ---

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
  };
  chart_data: Array<{ time: number; value: number }>;
  generatedId?: string;
}

const Icons = {
  Chart: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>,
  Server: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" /></svg>,
  Lock: () => <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>,
  Cpu: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>,
  Alert: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>,
  Code: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>,
  Download: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>,
  Save: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" /></svg>
};

const Skeleton = ({ className }: { className: string }) => (
  <div className={`animate-pulse bg-zinc-800/50 rounded ${className}`} />
);

const Badge = ({ children, type = "neutral" }: { children: React.ReactNode, type?: "neutral" | "success" | "warning" | "danger" | "info" }) => {
  const styles = {
    neutral: "bg-zinc-800 text-zinc-400 border-zinc-700",
    success: "bg-emerald-950/30 text-emerald-400 border-emerald-900/50",
    warning: "bg-amber-950/30 text-amber-400 border-amber-900/50",
    danger: "bg-rose-950/30 text-rose-400 border-rose-900/50",
    info: "bg-blue-950/30 text-blue-400 border-blue-900/50",
  };
  return <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border ${styles[type]}`}>{children}</span>;
};

// --- 2. MAIN COMPONENT ---

export default function Terminal() {
  const [loading, setLoading] = useState(false);
  const [strategy, setStrategy] = useState<StrategyResponse | null>(null);
  const [error, setError] = useState("");
  const [showCode, setShowCode] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");
  const [user, setUser] = useState<any>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  const router = useRouter();

  // Inputs
  const [symbol, setSymbol] = useState("BTC-USD");
  const [capital, setCapital] = useState(50000);
  const [minTrades, setMinTrades] = useState(25);
  const [maxTrades, setMaxTrades] = useState(150);
  const [targetPF, setTargetPF] = useState(1.5);
  const [riskMode, setRiskMode] = useState<"Funded" | "Live">("Funded");
  const [windowMode, setWindowMode] = useState("auto_shortest"); 
  const [isAssetOpen, setIsAssetOpen] = useState(false);

  const dashboardRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const assets = {
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"],
    "Stocks": ["TSLA", "AAPL", "NVDA", "MSFT", "AMD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
    "Metals": ["Gold", "Silver"]
  };

  // --- AUTH ---
  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => setUser(user));
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });
    return () => subscription.unsubscribe();
  }, []);

  const handleLogin = async () => {
    const email = prompt("Enter email for Magic Link login:");
    if (!email) return;
    const toastId = toast.loading("Sending magic link...");
    const { error } = await supabase.auth.signInWithOtp({ email });
    if (error) {
        toast.error(error.message, { id: toastId });
    } else {
        toast.success("Check your email! Magic link sent.", { id: toastId });
    }
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

    const { error } = await supabase.from('strategies').insert({
        user_id: user.id,
        symbol: symbol,
        name: strategy.strategy_name,
        entry_price: strategy.chart_data[strategy.chart_data.length-1]?.value || 100,
        pf: strategy.metrics.profit_factor,
        win_rate: 0,
        trades: strategy.metrics.total_txns,
        duration: strategy.metrics.duration,
        pine_code: strategy.metrics.pine_code
    });

    if (error) {
        toast.error("Error saving: " + error.message, { id: toastId });
    } else {
        toast.success("Strategy saved to Dashboard! 🚀", { id: toastId });
        setTimeout(() => router.push('/dashboard'), 1000);
    }
  };

  // --- EFFECTS ---
  useEffect(() => {
    const checkHealth = async () => {
        try {
            await fetch(process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000', { method: 'HEAD' });
            setBackendStatus("online");
        } catch (e) { setBackendStatus("offline"); }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) setIsAssetOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [dropdownRef]);

  const scrollToDashboard = () => dashboardRef.current?.scrollIntoView({ behavior: 'smooth' });

  // --- OPTIMIZATION LOGIC ---
  const runOptimization = async () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setStrategy(null);
    setError("");
    setShowCode(false);
    
    let cleanSymbol = symbol.split(' ')[0]; 
    if(symbol.includes("Gold")) cleanSymbol = "GC=F";
    if(symbol.includes("Silver")) cleanSymbol = "SI=F";

    const maxDrawdown = riskMode === "Funded" ? 10.0 : 30.0;

    try {
      const payload: OptimizationRequest = { 
        symbol: cleanSymbol, 
        initial_capital: capital, 
        min_trades: minTrades,
        max_trades: maxTrades,
        min_profit_factor: targetPF,
        max_drawdown: maxDrawdown,
        timeframe: "1h",
        window_mode: windowMode,
        window_days_candidates: [14, 21, 30, 45, 60, 90, 120, 180]
      };

      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000'}/generate-strategy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Optimization failed");
      
      setStrategy({
          ...data,
          generatedId: Math.random().toString(36).substr(2, 9).toUpperCase()
      });
    } catch (e: any) {
      if (e.name === 'AbortError') return;
      setError(e.message || "Server Offline. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
      if(!strategy) return;
      const headers = "Time,Equity\n";
      const rows = strategy.chart_data.map((d) => `${new Date(d.time*1000).toISOString()},${d.value}`).join("\n");
      const blob = new Blob([headers + rows], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${symbol}_strategy_data.csv`;
      a.click();
  };

  const getConfidenceLevel = (metrics: any) => {
      if (metrics.total_txns < minTrades + 5) return { label: "LOW DENSITY", type: "warning" as const };
      if (metrics.profit_factor < targetPF + 0.1) return { label: "MARGINAL", type: "warning" as const };
      if (metrics.max_dd > (riskMode === "Funded" ? 8 : 25)) return { label: "HIGH RISK", type: "danger" as const };
      return { label: "HIGH CONFIDENCE", type: "success" as const };
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500/30">
      
      {/* NAVBAR */}
      <nav className="fixed w-full z-50 bg-zinc-950/80 backdrop-blur-md border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            
            {/* LEFT SIDE: LOGO + TICKER */}
            <div className="flex items-center">
                <div className="text-xl font-bold tracking-tighter flex items-center gap-2 text-white mr-6">
                    <div className={`w-2 h-2 rounded-full animate-pulse shadow-[0_0_10px_currentColor] ${backendStatus === 'online' ? 'bg-emerald-500 text-emerald-500' : 'bg-red-500 text-red-500'}`}/>
                    LUX QUANT <span className="text-emerald-500">FACTORY</span>
                </div>
                
                {/* 🟢 LIVE TICKER */}
                <LiveTicker />
            </div>
            
            {/* RIGHT SIDE: BUTTONS */}
            <div className="flex items-center gap-4">
                {user ? (
                    <>
                        <button onClick={() => router.push('/dashboard')} className="text-xs text-zinc-400 hover:text-white transition-colors">DASHBOARD</button>
                        <button onClick={handleLogout} className="text-xs text-zinc-400 hover:text-white transition-colors">LOGOUT</button>
                    </>
                ) : (
                    <button onClick={handleLogin} className="text-xs text-zinc-400 hover:text-white transition-colors">LOGIN</button>
                )}
                <button onClick={scrollToDashboard} className="bg-zinc-100 text-zinc-950 px-5 py-2 rounded-full text-xs font-bold hover:bg-zinc-200 transition-colors border border-transparent hover:border-emerald-500">
                    LAUNCH TERMINAL
                </button>
            </div>
        </div>
      </nav>

      {/* HERO SECTION */}
      <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-6 overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-emerald-900/10 via-zinc-950 to-zinc-950 z-0"/>
          
          <div className="max-w-5xl mx-auto text-center relative z-10">
              <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/5 text-emerald-400 text-[10px] font-bold tracking-[0.2em] uppercase animate-in fade-in slide-in-from-bottom-4 duration-700">
                  <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full"/> v4.4 ECOSYSTEM LIVE
              </div>
              
              <h1 className="text-5xl md:text-8xl font-bold tracking-tighter mb-6 bg-gradient-to-b from-white via-white to-zinc-600 bg-clip-text text-transparent animate-in fade-in zoom-in-95 duration-1000">
                  Mathematically Proven <br/> Trading Alpha.
              </h1>
              
              <div className="flex flex-col md:flex-row gap-4 justify-center items-center animate-in fade-in slide-in-from-bottom-4 delay-300 duration-1000">
                  <button onClick={scrollToDashboard} className="group relative bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-4 rounded-xl text-sm font-bold transition-all w-full md:w-auto overflow-hidden shadow-lg shadow-emerald-900/20">
                      <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"/>
                      <span className="relative flex items-center gap-2 justify-center">START GENERATING ▼</span>
                  </button>
                  <p className="text-xs text-zinc-600 mt-4 md:mt-0">*Processing power required.<br/>Results may take 3-5 seconds.</p>
              </div>
          </div>
      </section>

      {/* DASHBOARD */}
      <section ref={dashboardRef} className="py-24 px-4 md:px-8 relative bg-zinc-950">
          <div className="max-w-7xl mx-auto">
              
              {/* HEADER & STATUS */}
              <div className="flex flex-col md:flex-row items-end justify-between border-b border-white/5 pb-6 mb-10 gap-4">
                  <div>
                      <h2 className="text-3xl font-bold text-white tracking-tight">Strategy Terminal</h2>
                      <p className="text-sm text-zinc-500 mt-1">Configure parameters. The AI hunts for the shortest valid validation window.</p>
                  </div>
                  <div className="flex gap-3">
                      <div className="flex flex-col items-end">
                          <span className="text-[10px] text-zinc-500 font-bold uppercase mb-1">System Status</span>
                          <Badge type={backendStatus === 'online' ? 'success' : 'danger'}>{backendStatus === 'online' ? 'ONLINE' : 'OFFLINE'}</Badge>
                      </div>
                      <div className="flex flex-col items-end">
                          <span className="text-[10px] text-zinc-500 font-bold uppercase mb-1">Data Feed</span>
                          <Badge type="info">YAHOO FINANCE API</Badge>
                      </div>
                  </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  
                  {/* SIDEBAR */}
                  <div className="lg:col-span-3 space-y-6 sticky top-24">
                      <div className="bg-zinc-900/50 p-6 rounded-2xl border border-white/5 backdrop-blur-sm">
                          <div className="space-y-6">
                              <div className="relative" ref={dropdownRef}>
                                  <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Asset Class</label>
                                  <button onClick={() => setIsAssetOpen(!isAssetOpen)} className="w-full mt-2 bg-zinc-950 border border-white/10 text-white py-3 px-4 rounded-lg flex justify-between items-center hover:border-emerald-500/50 transition-colors text-sm font-mono group">
                                      <span>{symbol}</span>
                                      <span className={`text-zinc-600 group-hover:text-emerald-500 transition-transform ${isAssetOpen ? 'rotate-180' : ''}`}>▼</span>
                                  </button>
                                  {isAssetOpen && (
                                      <div className="absolute z-50 w-full mt-2 bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl max-h-60 overflow-y-auto">
                                          {Object.entries(assets).map(([cat, items]) => (
                                              <div key={cat} className="p-2 border-b border-white/5 last:border-0">
                                                  <div className="text-[9px] text-zinc-500 font-bold px-2 py-1 uppercase">{cat}</div>
                                                  {items.map(item => (
                                                      <div key={item} onClick={() => { setSymbol(item); setIsAssetOpen(false); }} className="px-2 py-2 text-xs text-zinc-300 hover:bg-emerald-900/20 hover:text-emerald-400 rounded cursor-pointer font-mono transition-colors">{item}</div>
                                                  ))}
                                              </div>
                                          ))}
                                      </div>
                                  )}
                              </div>
                              <div>
                                  <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Initial Capital ($)</label>
                                  <input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))} className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-white font-mono text-sm focus:border-emerald-500/50 outline-none transition-all focus:ring-1 focus:ring-emerald-500/20" />
                              </div>
                              <div className="grid grid-cols-2 gap-3">
                                  <div>
                                      <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Min Trades</label>
                                      <input type="number" value={minTrades} onChange={(e) => setMinTrades(Number(e.target.value))} className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-center font-mono text-sm focus:border-blue-500/50 outline-none transition-all" />
                                  </div>
                                  <div>
                                      <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Max Trades</label>
                                      <input type="number" value={maxTrades} onChange={(e) => setMaxTrades(Number(e.target.value))} className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-center font-mono text-sm focus:border-blue-500/50 outline-none transition-all" />
                                  </div>
                              </div>
                              <div>
                                  <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Target PF</label>
                                  <input type="number" step="0.1" value={targetPF} onChange={(e) => setTargetPF(Number(e.target.value))} className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-center font-mono text-sm focus:border-emerald-500/50 outline-none transition-all" />
                              </div>
                              <div>
                                  <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Window Search</label>
                                  <select value={windowMode} onChange={(e) => setWindowMode(e.target.value)} className="w-full mt-2 bg-zinc-950 border border-white/10 p-3 rounded-lg text-white text-xs font-mono focus:border-purple-500/50 outline-none appearance-none">
                                      <option value="auto_shortest">Auto (Find Shortest Valid)</option>
                                      <option value="multi_recent">Stability (Check Last 3)</option>
                                  </select>
                              </div>
                              <div>
                                  <label className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Risk Mode</label>
                                  <div className="grid grid-cols-2 gap-2 mt-2">
                                      <button onClick={() => setRiskMode("Funded")} className={`py-3 rounded-lg text-[10px] font-bold transition-all border ${riskMode === "Funded" ? "bg-emerald-900/20 border-emerald-500 text-emerald-400" : "bg-zinc-950 border-white/10 text-zinc-500 hover:border-zinc-700"}`}>FUNDED (10%)</button>
                                      <button onClick={() => setRiskMode("Live")} className={`py-3 rounded-lg text-[10px] font-bold transition-all border ${riskMode === "Live" ? "bg-blue-900/20 border-blue-500 text-blue-400" : "bg-zinc-950 border-white/10 text-zinc-500 hover:border-zinc-700"}`}>LIVE (30%)</button>
                                  </div>
                              </div>
                              <button onClick={runOptimization} disabled={loading} className="w-full bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white py-4 rounded-xl text-sm font-bold transition-all shadow-lg shadow-emerald-900/20 disabled:opacity-50 disabled:cursor-not-allowed mt-4 relative overflow-hidden group">
                                {loading ? (
                                    <div className="flex items-center justify-center gap-3">
                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"/> 
                                        <span className="tracking-wide text-xs">CRUNCHING...</span>
                                    </div>
                                ) : (
                                    <span className="group-hover:tracking-widest transition-all duration-300">FIND ALPHA</span>
                                )}
                              </button>
                          </div>
                      </div>
                  </div>

                  {/* MAIN DISPLAY */}
                  <div className="lg:col-span-9 space-y-6">
                      
                      {error && (
                          <div className="bg-rose-950/10 border border-rose-900/30 text-rose-400 p-6 rounded-2xl flex items-center gap-4 animate-in fade-in slide-in-from-top-2">
                              <Icons.Alert />
                              <div><p className="text-sm font-bold">Optimization Failed</p><p className="text-xs opacity-80 mt-1 font-mono">{error}</p></div>
                          </div>
                      )}

                      {!strategy && !error && !loading && (
                          <div className="h-full min-h-[500px] flex flex-col items-center justify-center text-zinc-600 border border-dashed border-white/10 rounded-2xl bg-zinc-900/20">
                              <div className="w-20 h-20 mb-6 rounded-2xl bg-zinc-950 flex items-center justify-center shadow-2xl shadow-black border border-white/5"><Icons.Lock /></div>
                              <p className="text-sm font-bold tracking-widest uppercase text-zinc-500">System Ready</p>
                              <p className="text-xs mt-2 opacity-40 max-w-xs text-center">Select an asset and click 'Find Alpha' to initiate the quantitative analysis engine.</p>
                          </div>
                      )}

                      {loading && (
                        <div className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">{[1,2,3,4].map(i => <Skeleton key={i} className="h-32 w-full rounded-xl" />)}</div>
                            <Skeleton className="h-[500px] w-full rounded-2xl" />
                        </div>
                      )}

                      {strategy && !loading && (
                          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6">
                              <div className="bg-gradient-to-r from-zinc-900 to-zinc-950 border border-white/10 p-6 rounded-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                                  <div>
                                      <h3 className="text-lg font-bold text-white flex items-center gap-2">Strategy Found: <span className="text-emerald-400">{strategy.strategy_name}</span></h3>
                                      <p className="text-sm text-zinc-400 mt-1 max-w-2xl">The engine identified a statistical edge in the <strong>last {strategy.window_days} days</strong>. It executes approximately <span className="text-zinc-200 font-mono">{(strategy.metrics.total_txns / (strategy.window_days || 1) * 7).toFixed(1)} trades/week</span>.</p>
                                  </div>
                                  <div className="flex flex-col items-end gap-2"><Badge type={getConfidenceLevel(strategy.metrics).type}>{getConfidenceLevel(strategy.metrics).label}</Badge><span className="text-[10px] text-zinc-600 font-mono">ID: {strategy.generatedId || "GEN-001"}</span></div>
                              </div>

                              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                  <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-sm relative overflow-hidden group">
                                      <div className="absolute top-0 right-0 w-20 h-20 bg-emerald-500/5 rounded-full blur-2xl group-hover:bg-emerald-500/10 transition-colors"/>
                                      <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Net Profit</p>
                                      <p className={`text-2xl font-mono font-bold ${strategy.metrics.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>{strategy.metrics.total_pnl >= 0 ? '+' : ''}${strategy.metrics.total_pnl.toLocaleString(undefined, {maximumFractionDigits: 0})}</p>
                                      <p className="text-xs text-zinc-500 mt-1 font-mono">{strategy.metrics.pnl_percent.toFixed(2)}% Return</p>
                                  </div>
                                  <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-sm">
                                      <div className="flex justify-between items-start">
                                          <div>
                                              <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Profit Factor</p>
                                              <p className={`text-2xl font-mono font-bold ${strategy.metrics.profit_factor >= targetPF ? 'text-emerald-400' : 'text-amber-500'}`}>{strategy.metrics.profit_factor}</p>
                                          </div>
                                          <div className="text-[10px] bg-zinc-950 px-2 py-1 rounded text-zinc-400 border border-white/5">{strategy.metrics.note}</div>
                                      </div>
                                  </div>
                                  <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-sm">
                                      <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Drawdown</p>
                                      <p className={`text-2xl font-mono font-bold ${strategy.metrics.max_dd > (riskMode==="Funded"?9:25) ? "text-rose-400" : "text-zinc-200"}`}>{strategy.metrics.max_dd}%</p>
                                      <p className="text-xs text-zinc-500 mt-1">Peak-to-Valley Risk</p>
                                  </div>
                                  <div className="bg-zinc-900/50 p-5 rounded-xl border border-white/5 backdrop-blur-sm">
                                      <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider mb-2">Volume</p>
                                      <div className="flex items-baseline gap-2">
                                          <p className="text-2xl font-mono font-bold text-blue-400">{strategy.metrics.total_txns}</p>
                                          <span className="text-xs text-zinc-500">Trades</span>
                                      </div>
                                      <p className="text-xs text-zinc-500 mt-1 font-mono">{strategy.metrics.duration}</p>
                                  </div>
                              </div>

                              <div className="bg-zinc-900/50 p-6 rounded-2xl border border-white/5 min-h-[500px] flex flex-col shadow-2xl backdrop-blur-sm">
                                  <div className="flex justify-between items-center mb-6">
                                      <div><h3 className="font-bold text-white text-lg flex items-center gap-2">Equity Curve<Badge type="info">{symbol}</Badge></h3><p className="text-xs text-zinc-500 font-mono mt-1">Simulated performance. Static view.</p></div>
                                      <div className="flex gap-2">
                                          <button onClick={saveStrategy} className="flex items-center gap-2 text-[10px] bg-emerald-950/30 text-emerald-400 hover:bg-emerald-900/50 px-3 py-1.5 rounded border border-emerald-900/50 transition-colors font-bold"><Icons.Save /> SAVE</button>
                                          <button onClick={downloadCSV} className="flex items-center gap-2 text-[10px] bg-zinc-950 hover:bg-zinc-800 text-zinc-300 px-3 py-1.5 rounded border border-white/10 transition-colors font-bold"><Icons.Download /> CSV</button>
                                          <button onClick={() => setShowCode(!showCode)} className="flex items-center gap-2 text-[10px] bg-emerald-950/30 text-emerald-500 hover:bg-emerald-950/50 px-3 py-1.5 rounded border border-emerald-900/50 transition-colors font-bold"><Icons.Code /> {showCode ? "HIDE PINE" : "VIEW PINE"}</button>
                                      </div>
                                  </div>
                                  {showCode ? (
                                      <div className="flex-grow bg-zinc-950 border border-white/5 rounded-xl overflow-hidden relative group animate-in fade-in slide-in-from-right-4">
                                          <div className="bg-zinc-900 px-4 py-3 flex justify-between items-center border-b border-white/5">
                                               <span className="text-[10px] text-zinc-400 font-bold uppercase tracking-wider flex items-center gap-2">TradingView Pine Script v5</span>
                                               <button onClick={() => navigator.clipboard.writeText(strategy.metrics.pine_code)} className="text-[10px] bg-zinc-100 text-black px-3 py-1 rounded hover:bg-zinc-300 font-bold transition-colors">COPY CODE</button>
                                          </div>
                                          <textarea readOnly value={strategy.metrics.pine_code} className="w-full h-full min-h-[400px] bg-transparent text-xs font-mono text-emerald-400/90 p-4 outline-none resize-none"/>
                                      </div>
                                  ) : (
                                      <div className="flex-grow w-full relative border border-white/5 rounded-xl overflow-hidden bg-zinc-950/50">
                                            <div className="absolute inset-0 pointer-events-none">
                                                <StrategyChart data={strategy.chart_data} />
                                            </div>
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
      <footer className="py-12 border-t border-white/5 bg-zinc-950">
          <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
              <div className="text-xs text-zinc-600">&copy; {new Date().getFullYear()} Lux Quant AI. Not financial advice. Data provided for educational purposes.</div>
              <div className="flex gap-6 text-xs text-zinc-500 font-bold uppercase tracking-wider"><a href="#" className="hover:text-white transition-colors">Documentation</a><a href="#" className="hover:text-white transition-colors">Terms</a><a href="#" className="hover:text-white transition-colors">Privacy</a></div>
          </div>
      </footer>
    </main>
  );
}
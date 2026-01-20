"use client";
import { useEffect, useState } from 'react';
import { supabase } from '../supabase';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

export default function Dashboard() {
  const [strategies, setStrategies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push('/'); 
      } else {
        fetchStrategies(user.id);
      }
    };
    checkUser();
  }, [router]);

  const fetchStrategies = async (userId: string) => {
    const { data, error } = await supabase
      .from('strategies')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });
    
    if (error) {
        toast.error("Failed to sync with neural archives");
    } else {
        setStrategies(data || []);
    }
    setLoading(false);
  };

  const deleteStrategy = async (id: string) => {
    const { error } = await supabase.from('strategies').delete().eq('id', id);
    if (error) {
        toast.error("Execution failed: Could not close strategy");
    } else {
        toast.success("Strategy terminated successfully");
        setStrategies(prev => prev.filter(s => s.id !== id));
    }
  };

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    toast.success("Pine Script copied to clipboard 📋");
  };

  // Simulation: Random market move (+/- 5%)
  const getPaperPnL = (entryPrice: number) => {
    const randomMove = 1 + (Math.random() * 0.1 - 0.05); 
    const currentPrice = entryPrice * randomMove;
    return ((currentPrice - entryPrice) / entryPrice) * 100;
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500/30 relative overflow-x-hidden">
      
      {/* 1. BACKGROUND GRID (Matches Terminal) */}
      <div className="fixed inset-0 z-0 pointer-events-none">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_800px_at_50%_200px,#00000000,transparent)]"></div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 relative z-10">
        
        {/* NAV HEADER */}
        <nav className="flex justify-between items-end mb-12 border-b border-white/5 pb-8">
          <div>
            <div className="inline-flex items-center gap-2 mb-2 px-3 py-1 rounded-full border border-emerald-500/20 bg-emerald-500/5 text-emerald-400 text-[9px] font-bold tracking-widest uppercase">
               <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse"/> Verified Archives
            </div>
            <h1 className="text-4xl font-bold text-white tracking-tighter">Command <span className="text-emerald-500">Center</span></h1>
            <p className="text-zinc-500 text-xs mt-2 font-mono">Secured connection to proprietary alpha repository.</p>
          </div>
          <button 
            onClick={() => router.push('/')} 
            className="group flex items-center gap-2 text-xs font-bold text-zinc-400 hover:text-white transition-colors border border-white/5 px-4 py-2 rounded-xl bg-zinc-900/50 backdrop-blur-sm"
          >
            <span className="group-hover:-translate-x-1 transition-transform">←</span> TERMINAL
          </button>
        </nav>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-40 gap-4">
             <div className="w-8 h-8 border-2 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin"/>
             <p className="text-zinc-500 font-mono text-[10px] uppercase tracking-widest">Decrypting Neural Data...</p>
          </div>
        ) : (
          <div className="space-y-6">
            {strategies.length === 0 ? (
              <div className="text-center py-32 border border-dashed border-white/10 rounded-3xl bg-zinc-900/20 backdrop-blur-sm">
                <p className="text-zinc-500 font-mono text-sm">Vault is currently empty.</p>
                <button 
                    onClick={() => router.push('/')} 
                    className="mt-6 bg-white text-black hover:bg-emerald-500 hover:text-white px-8 py-3 rounded-full text-xs font-bold transition-all"
                >
                    INITIATE SEARCH
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {strategies.map((s) => {
                  const pnl = getPaperPnL(s.entry_price || 100);
                  return (
                    <div key={s.id} className="group relative bg-zinc-900/40 border border-white/5 p-6 rounded-2xl flex flex-col md:flex-row items-center justify-between gap-6 hover:border-emerald-500/30 transition-all duration-500 hover:shadow-[0_0_40px_-15px_rgba(16,185,129,0.2)]">
                      
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <span className="text-xl font-bold text-white tracking-tight">{s.symbol}</span>
                          <span className="text-[10px] font-bold uppercase tracking-widest bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">{s.name}</span>
                        </div>
                        <div className="flex flex-wrap gap-6 text-[10px] text-zinc-500 font-mono uppercase">
                          <div className="flex flex-col">
                            <span className="text-zinc-600 text-[8px] font-bold">Profit Factor</span>
                            <span className="text-zinc-300">{s.pf}</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-zinc-600 text-[8px] font-bold">Total Trades</span>
                            <span className="text-zinc-300">{s.trades}</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-zinc-600 text-[8px] font-bold">Deployed On</span>
                            <span className="text-zinc-300">{new Date(s.created_at).toLocaleDateString()}</span>
                          </div>
                        </div>
                      </div>

                      <div className="px-8 border-x border-white/5 text-center hidden md:block">
                        <div className="text-[9px] uppercase font-bold text-zinc-600 mb-1 tracking-tighter">Live Paper Simulation</div>
                        <div className={`text-2xl font-mono font-bold tabular-nums ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <button 
                            onClick={() => copyCode(s.pine_code)} 
                            className="h-11 px-6 bg-zinc-950 border border-white/10 rounded-xl text-[10px] font-bold uppercase tracking-widest hover:bg-zinc-800 transition-colors"
                        >
                            Copy Code
                        </button>
                        <button 
                            onClick={() => deleteStrategy(s.id)} 
                            className="h-11 px-4 bg-rose-950/10 border border-rose-900/20 text-rose-500 rounded-xl text-[10px] font-bold uppercase hover:bg-rose-500 hover:text-white transition-all"
                        >
                            Terminate
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="max-w-6xl mx-auto px-6 py-12 text-[10px] text-zinc-600 font-mono uppercase tracking-[0.2em] text-center opacity-50">
          Lux Quant AI // Secure Asset Repository // v4.4
      </footer>
    </main>
  );
}
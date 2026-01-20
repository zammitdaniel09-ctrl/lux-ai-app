"use client";
import { useEffect, useState } from 'react';
import { supabase } from '../supabase';
import { useRouter } from 'next/navigation';

export default function Dashboard() {
  const [strategies, setStrategies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const checkUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push('/'); // Redirect if not logged in
      } else {
        fetchStrategies();
      }
    };
    checkUser();
  }, [router]);

  const fetchStrategies = async () => {
    const { data } = await supabase
      .from('strategies')
      .select('*')
      .order('created_at', { ascending: false });
    
    if (data) setStrategies(data);
    setLoading(false);
  };

  const deleteStrategy = async (id: string) => {
    await supabase.from('strategies').delete().eq('id', id);
    fetchStrategies(); // Refresh list
  };

  // Simulation: Random market move (+/- 5%) to show "Paper Trading" effect
  const getPaperPnL = (entryPrice: number) => {
    const randomMove = 1 + (Math.random() * 0.1 - 0.05); 
    const currentPrice = entryPrice * randomMove;
    return ((currentPrice - entryPrice) / entryPrice) * 100;
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 p-8 font-sans">
      <nav className="flex justify-between items-center mb-12 border-b border-white/5 pb-6">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"/>
          Live Dashboard
        </h1>
        <button onClick={() => router.push('/')} className="text-xs text-zinc-500 hover:text-white">
          ← Back to Generator
        </button>
      </nav>

      {loading ? (
        <div className="text-center text-zinc-500 mt-20">Loading your alpha...</div>
      ) : (
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 gap-6">
            {strategies.length === 0 ? (
              <div className="text-center py-20 border border-dashed border-zinc-800 rounded-xl">
                <p className="text-zinc-500">No active strategies.</p>
                <button onClick={() => router.push('/')} className="mt-4 bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm">Create One</button>
              </div>
            ) : (
              strategies.map((s) => {
                const pnl = getPaperPnL(s.entry_price || 100);
                return (
                  <div key={s.id} className="bg-zinc-900/50 border border-white/5 p-6 rounded-xl flex flex-col md:flex-row items-center justify-between gap-6 hover:border-white/10 transition-colors">
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <span className="text-lg font-bold text-white">{s.symbol}</span>
                        <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded border border-zinc-700">{s.name}</span>
                      </div>
                      <div className="flex gap-4 text-xs text-zinc-500 font-mono">
                        <span>PF: {s.pf}</span>
                        <span>Txns: {s.trades}</span>
                        <span>Created: {new Date(s.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-[10px] uppercase font-bold text-zinc-600 mb-1">Simulated PnL</div>
                      <div className={`text-xl font-mono font-bold ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <button onClick={() => navigator.clipboard.writeText(s.pine_code)} className="px-4 py-2 bg-zinc-950 border border-white/10 rounded-lg text-xs font-bold hover:bg-zinc-800">Copy Code</button>
                      <button onClick={() => deleteStrategy(s.id)} className="px-4 py-2 bg-rose-950/20 border border-rose-900/30 text-rose-400 rounded-lg text-xs font-bold hover:bg-rose-950/40">Close</button>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      )}
    </main>
  );
}
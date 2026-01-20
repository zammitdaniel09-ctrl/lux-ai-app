"use client";
import React from 'react';

export const MarketWidget = () => {
  return (
    <div className="bg-zinc-900/50 p-5 rounded-2xl border border-white/5 backdrop-blur-sm space-y-5">
      <div className="flex items-center justify-between border-b border-white/5 pb-3">
         <span className="text-[10px] uppercase font-bold text-zinc-500 tracking-wider">Market Vitals</span>
         <div className="flex items-center gap-2">
             <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"/>
             <span className="text-[9px] text-emerald-500 font-bold">LIVE</span>
         </div>
      </div>
      
      {/* VIX */}
      <div>
        <div className="flex justify-between text-xs mb-2 font-mono">
          <span className="text-zinc-400">VIX (Volatility)</span>
          <span className="text-emerald-400">14.2</span>
        </div>
        <div className="w-full bg-zinc-950 h-1.5 rounded-full overflow-hidden border border-white/5">
          <div className="bg-emerald-500 h-full w-[30%] shadow-[0_0_10px_#10b981]"/>
        </div>
      </div>

      {/* Sentiment */}
      <div>
        <div className="flex justify-between text-xs mb-2 font-mono">
          <span className="text-zinc-400">Sentiment</span>
          <span className="text-blue-400">Greed (72)</span>
        </div>
        <div className="w-full bg-zinc-950 h-1.5 rounded-full overflow-hidden border border-white/5">
          <div className="bg-blue-500 h-full w-[72%] shadow-[0_0_10px_#3b82f6]"/>
        </div>
      </div>

      {/* AI Load */}
      <div>
        <div className="flex justify-between text-xs mb-2 font-mono">
          <span className="text-zinc-400">AI Server Load</span>
          <span className="text-amber-400">42%</span>
        </div>
        <div className="w-full bg-zinc-950 h-1.5 rounded-full overflow-hidden border border-white/5">
          <div className="bg-amber-500 h-full w-[42%] shadow-[0_0_10px_#f59e0b]"/>
        </div>
      </div>
    </div>
  );
};
"use client";

import React, { useState, useEffect, useRef } from 'react';

// Define messages outside the component to keep the reference stable
const MESSAGES = [
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

export const TerminalLog = ({ active }: { active: boolean }) => {
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!active) return;

    setLogs([]); // Reset logs when activation starts
    let i = 0;

    const interval = setInterval(() => {
      // Use the external MESSAGES constant
      if (i < MESSAGES.length) {
        const msg = MESSAGES[i];
        setLogs(prev => [...prev, `> ${msg}`]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 600); 

    return () => clearInterval(interval);
  }, [active]);

  // Auto-scroll to bottom whenever logs update
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  if (!active) return null;

  return (
    <div className="font-mono text-xs text-emerald-500/90 bg-black/80 p-6 rounded-xl border border-emerald-500/20 h-64 overflow-y-auto mb-6 backdrop-blur-md shadow-[inset_0_0_20px_rgba(16,185,129,0.1)]">
      <div className="flex flex-col gap-2">
        {logs.map((log, idx) => (
          <div key={idx} className="animate-in fade-in slide-in-from-left-2 duration-300">
            {log} <span className="text-emerald-800 ml-2">[OK]</span>
          </div>
        ))}
        <div ref={logsEndRef} />
        {/* Blinking cursor while running */}
        {logs.length < MESSAGES.length && (
             <div className="animate-pulse text-emerald-400">_</div>
        )}
      </div>
    </div>
  );
};
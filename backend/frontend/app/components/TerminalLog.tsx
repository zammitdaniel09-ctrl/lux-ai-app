"use client";
import React, { useState, useEffect, useRef } from 'react';

export const TerminalLog = ({ active }: { active: boolean }) => {
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // The "script" the AI will type out
  const messages = [
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

  useEffect(() => {
    if (active) {
      setLogs([]); // Clear logs on start
      let i = 0;
      const interval = setInterval(() => {
        if (i < messages.length) {
            const msg = messages[i];
            setLogs(prev => [...prev, `> ${msg}`]);
            i++;
        } else {
          clearInterval(interval);
        }
      }, 600); // Add a new line every 600ms
      return () => clearInterval(interval);
    }
  }, [active]);

  // Auto-scroll to bottom
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
        {logs.length < messages.length && (
             <div className="animate-pulse text-emerald-400">_</div>
        )}
      </div>
    </div>
  );
};
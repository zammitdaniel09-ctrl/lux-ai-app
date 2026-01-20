"use client";
import React, { useState, useEffect, useRef } from 'react';
import useWebSocket from 'react-use-websocket';

interface BinanceTickerData {
  s: string;
  p: string;
  [key: string]: any;
}

const SOCKET_URL = 'wss://stream.binance.com:9443/ws/btcusdt@trade/ethusdt@trade/solusdt@trade';

export const LiveTicker = () => {
  const [prices, setPrices] = useState({ BTC: 0, ETH: 0, SOL: 0 });
  const lastUpdateTime = useRef(0);

  // FIXED: Removed the 'throttle' property that was causing the error
  const { lastJsonMessage } = useWebSocket<BinanceTickerData>(SOCKET_URL, {
    shouldReconnect: () => true,
  });

  useEffect(() => {
    if (lastJsonMessage && lastJsonMessage.s && lastJsonMessage.p) {
      const now = Date.now();
      // Manual Throttling: Only update once per second
      if (now - lastUpdateTime.current > 1000) {
        const symbol = lastJsonMessage.s.replace('USDT', '');
        const price = parseFloat(lastJsonMessage.p);
        setPrices(prev => ({ ...prev, [symbol]: price }));
        lastUpdateTime.current = now;
      }
    }
  }, [lastJsonMessage]);

  return (
    <div className="hidden md:flex items-center gap-6 text-[10px] font-mono text-zinc-500 border-l border-white/10 pl-6 ml-6 h-8">
      {Object.entries(prices).map(([coin, price]) => (
        <div key={coin} className="flex gap-2">
          <span className="font-bold text-zinc-400">{coin}</span>
          <span className={price > 0 ? "text-emerald-400" : "text-zinc-600"}>
            ${price > 0 ? price.toFixed(2) : "Loading..."}
          </span>
        </div>
      ))}
      <div className="flex items-center gap-1">
        <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"/>
        <span className="text-emerald-500 font-bold">LIVE FEED</span>
      </div>
    </div>
  );
};
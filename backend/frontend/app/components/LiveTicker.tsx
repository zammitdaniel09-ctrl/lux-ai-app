"use client";
import React, { useState, useEffect } from 'react';
import useWebSocket from 'react-use-websocket';

// 1. Define the shape of the Binance stream data
interface BinanceTickerData {
  s: string; // Symbol (e.g., "BTCUSDT")
  p: string; // Price (e.g., "93000.50")
  [key: string]: any; // Allow other properties just in case
}

export const LiveTicker = () => {
  const [prices, setPrices] = useState({ BTC: 0, ETH: 0, SOL: 0 });
  
  // Connect to Binance Public Stream
  const socketUrl = 'wss://stream.binance.com:9443/ws/btcusdt@trade/ethusdt@trade/solusdt@trade';

  // 2. Pass the interface <BinanceTickerData> to the hook
  const { lastJsonMessage } = useWebSocket<BinanceTickerData>(socketUrl, {
    shouldReconnect: () => true,
  });

  useEffect(() => {
    // 3. Now TypeScript knows 's' and 'p' exist on lastJsonMessage
    if (lastJsonMessage && lastJsonMessage.s && lastJsonMessage.p) {
      const symbol = lastJsonMessage.s.replace('USDT', '');
      const price = parseFloat(lastJsonMessage.p);
      
      // Update the specific coin price
      setPrices(prev => ({ ...prev, [symbol]: price }));
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
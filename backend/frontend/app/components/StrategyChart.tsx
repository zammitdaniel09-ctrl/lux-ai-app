"use client";
import React, { useEffect, useRef } from "react";
import { createChart, ColorType, AreaSeries } from "lightweight-charts";

export const StrategyChart = ({ data }: any) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!data || !chartContainerRef.current) return;
    
    // 1. Create Chart with "Clean" Look
    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#9ca3af" },
      width: chartContainerRef.current.clientWidth,
      height: 350,
      grid: { vertLines: { visible: false }, horzLines: { color: "#1f2937", style: 3 } }, // Minimal grid
      timeScale: { borderColor: "#374151", timeVisible: true },
      rightPriceScale: { borderVisible: false },
    });

    // 2. Add GLOWING Area Series
    const areaSeries = chart.addSeries(AreaSeries, { 
      lineColor: '#10b981', // Emerald Green Line
      topColor: 'rgba(16, 185, 129, 0.4)', // Gradient Top
      bottomColor: 'rgba(16, 185, 129, 0.0)', // Transparent Bottom
      lineWidth: 2,
    });
    
    areaSeries.setData(data);
    chart.timeScale().fitContent();

    // Handle Resize
    const handleResize = () => {
        if(chartContainerRef.current) {
            chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        }
    };
    window.addEventListener('resize', handleResize);

    return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
    };
  }, [data]);

  return <div ref={chartContainerRef} className="w-full" />;
};
"use client";
import React from 'react';

interface LegalModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  content: string;
}

export const LegalModal = ({ isOpen, onClose, title, content }: LegalModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-zinc-950 border border-white/10 w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden relative">
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-white/10 bg-zinc-900/50">
            <h3 className="text-xl font-bold text-white tracking-tight">{title}</h3>
            <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors text-xl">&times;</button>
        </div>
        
        {/* Scrollable Content */}
        <div className="p-8 max-h-[60vh] overflow-y-auto text-zinc-400 text-sm leading-relaxed space-y-4 font-mono">
            {content.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
            ))}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/10 bg-zinc-900/50 flex justify-end">
            <button onClick={onClose} className="bg-white text-black px-6 py-2 rounded-full text-xs font-bold hover:bg-emerald-400 transition-colors">
                ACKNOWLEDGE
            </button>
        </div>
      </div>
    </div>
  );
};
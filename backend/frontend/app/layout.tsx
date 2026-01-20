import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
// LINE 5: IMPORT THIS
import { Toaster } from 'sonner';

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Lux Quant AI",
  description: "Institutional Grade Strategy Generator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {/* LINE 22: THIS MUST BE HERE OR TOASTS WON'T SHOW */}
        <Toaster richColors theme="dark" position="top-center" />
        
        {children}
      </body>
    </html>
  );
}
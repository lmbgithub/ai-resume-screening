import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Resume Screen",
  description: "Match a resume against a job description with a local Ollama model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}

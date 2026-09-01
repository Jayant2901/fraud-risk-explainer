import { useState } from "react";
import LiveScoring from "./components/LiveScoring";
import CostAnalysis from "./components/CostAnalysis";

type Tab = "live" | "cost";

export default function App() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-5">
        <h1 className="text-2xl font-bold">AI Risk Manager</h1>
        <p className="text-sm text-slate-400 mt-1">
          Entity-aware transaction risk scoring with a local LLM reasoning layer — Razorpay
          Buildathon, Track 2
        </p>
      </header>

      <nav className="flex gap-1 px-6 pt-4 border-b border-slate-800">
        {[
          { id: "live" as const, label: "Live Scoring" },
          { id: "cost" as const, label: "Cost-Optimal Threshold" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t-md border-b-2 transition ${
              tab === t.id
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="p-6">
        {tab === "live" ? <LiveScoring /> : <CostAnalysis />}
      </main>
    </div>
  );
}

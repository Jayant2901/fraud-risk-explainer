import { useState, type ComponentType } from "react";
import Overview from "./components/Overview";
import LiveScoring from "./components/LiveScoring";
import CostAnalysis from "./components/CostAnalysis";
import ReviewQueue from "./components/ReviewQueue";
import ConsistencyAnalysis from "./components/ConsistencyAnalysis";
import { focusRing, typeScale } from "./theme";

type Tab = "overview" | "live" | "cost" | "review" | "consistency";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "live", label: "Live Scoring" },
  { id: "review", label: "Review Queue" },
  { id: "cost", label: "Cost-Optimal Threshold" },
  { id: "consistency", label: "Consistency" },
];

const TAB_PANELS: Record<Tab, ComponentType> = {
  overview: Overview,
  live: LiveScoring,
  review: ReviewQueue,
  cost: CostAnalysis,
  consistency: ConsistencyAnalysis,
};

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const ActivePanel = TAB_PANELS[tab];

  return (
    <div className="min-h-screen bg-app-bg text-app-ink">
      <header className="border-b border-app-rule px-6 py-5">
        <h1 className={typeScale.pageTitle}>AI Risk Manager</h1>
        <p className={`${typeScale.caption} mt-1`}>
          Entity-aware transaction risk scoring with a local LLM reasoning layer — Razorpay
          Buildathon, Track 2
        </p>
      </header>

      <nav className="flex gap-1 px-6 pt-4 border-b border-app-rule" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t-md border-b-2 transition ${focusRing} ${
              tab === t.id
                ? "border-app-accent text-app-accent-soft"
                : "border-transparent text-app-faint hover:text-app-muted"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="p-6">
        <div className="max-w-7xl mx-auto">
          <ActivePanel />
        </div>
      </main>
    </div>
  );
}

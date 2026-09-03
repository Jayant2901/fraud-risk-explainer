import { useState, type ComponentType } from "react";
import Overview from "./components/Overview";
import LiveScoring from "./components/LiveScoring";
import ReviewQueue from "./components/ReviewQueue";
import ModelValidation from "./components/ModelValidation";
import SegmentedControl from "./components/SegmentedControl";
import { surfaceTranslucent, typeScale } from "./theme";

type Tab = "overview" | "live" | "review" | "validation";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "live", label: "Live Scoring" },
  { id: "review", label: "Review Queue" },
  { id: "validation", label: "Model Validation" },
];

const TAB_PANELS: Record<Tab, ComponentType> = {
  overview: Overview,
  live: LiveScoring,
  review: ReviewQueue,
  validation: ModelValidation,
};

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const ActivePanel = TAB_PANELS[tab];

  return (
    <div className="min-h-screen bg-app-bg text-app-ink">
      <header
        className={`sticky top-0 z-10 border-b border-app-rule px-6 py-4 ${surfaceTranslucent}`}
      >
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className={typeScale.pageTitle}>AI Risk Manager</h1>
            <p className={`${typeScale.caption} mt-1`}>
              Entity-aware transaction risk scoring with a local LLM reasoning layer — Razorpay
              Buildathon, Track 2
            </p>
          </div>
          <SegmentedControl
            segments={TABS}
            activeId={tab}
            onChange={(id) => setTab(id as Tab)}
            ariaLabel="Sections"
          />
        </div>
      </header>

      <main className="p-6">
        {/* Keyed on the tab id so React remounts on switch and the entrance
            animation replays. Nothing is queued or blocked: a fast
            double-click just mounts the newest panel, and the CSS
            animation is skipped entirely under reduced motion. */}
        <div key={tab} className="max-w-7xl mx-auto animate-panel-in">
          <ActivePanel />
        </div>
      </main>
    </div>
  );
}

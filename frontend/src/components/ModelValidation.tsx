import { useState, type ComponentType } from "react";
import CostAnalysis from "./CostAnalysis";
import EscalationAblation from "./EscalationAblation";
import ColdStartAnalysis from "./ColdStartAnalysis";
import ConsistencyAnalysis from "./ConsistencyAnalysis";
import ShadowScoring from "./ShadowScoring";
import { focusRing, typeScale } from "../theme";

type SectionId = "cost" | "escalation" | "cold-start" | "consistency" | "shadow";

const SECTIONS: { id: SectionId; label: string; Panel: ComponentType }[] = [
  { id: "cost", label: "Cost-Optimal Threshold, Sensitivity & Drift", Panel: CostAnalysis },
  { id: "escalation", label: "Entity Escalation Ablation", Panel: EscalationAblation },
  { id: "cold-start", label: "Cold-Start Graph Features", Panel: ColdStartAnalysis },
  { id: "consistency", label: "Consistency", Panel: ConsistencyAnalysis },
  { id: "shadow", label: "Shadow Model Comparison", Panel: ShadowScoring },
];

// Every analysis this app can show, in one place — nothing here is new
// content, just CostAnalysis.tsx/ConsistencyAnalysis.tsx (unchanged)
// plus two small new report-reading panels, nested as an accordion so
// Live Scoring and Review Queue can be the first two tabs a judge sees.
export default function ModelValidation() {
  const [openSections, setOpenSections] = useState<Set<SectionId>>(new Set(["cost"]));

  function toggle(id: SectionId) {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className={typeScale.sectionTitle}>Model Validation</h2>
        <p className={`${typeScale.body} mt-1 max-w-2xl`}>
          Every offline analysis this project ran to check its own claims, in one place — cost
          modeling, entity escalation, cold-start coverage, temporal drift, LLM
          self-consistency, and live shadow-model comparison. Nothing here is deleted or hidden,
          just consolidated so Live Scoring and Review Queue can stay the two tabs you see first.
        </p>
      </div>

      <div className="border border-app-rule rounded-xl divide-y divide-app-rule">
        {SECTIONS.map(({ id, label, Panel }) => {
          const isOpen = openSections.has(id);
          return (
            <div key={id}>
              <button
                onClick={() => toggle(id)}
                aria-expanded={isOpen}
                className={`w-full flex items-center justify-between px-4 py-3 text-left text-sm font-medium text-app-ink hover:bg-app-ink/5 ${focusRing}`}
              >
                {label}
                <span className="text-app-faint text-xs">{isOpen ? "▲" : "▼"}</span>
              </button>
              {isOpen && (
                <div className="px-4 pb-5">
                  <Panel />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

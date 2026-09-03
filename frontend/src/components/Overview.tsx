import { useEffect, useState } from "react";
import { api, type CostAnalysis } from "../api/client";
import RiskGauge from "./RiskGauge";
import { useAnimatedNumber } from "../hooks";
import {
  groupedRow,
  groupedRowLabel,
  groupedRowValue,
  noticeClass,
  surface,
  typeScale,
} from "../theme";

const TAB_QUESTIONS: { label: string; question: string }[] = [
  { label: "Live Scoring", question: "Given one transaction and an entity's recent history, what would the model + LLM reviewer decide right now?" },
  { label: "Review Queue", question: "When a human disposes flagged verdicts as fraud or false-positive, how precise is the model — overall, and specifically when it escalated?" },
];

const VALIDATION_CHIPS = [
  "Cost sensitivity",
  "Escalation ablation",
  "Temporal drift",
  "Cold-start graph signal",
  "Consistency",
];

// The gauge's starting position — deliberately just under the REVIEW
// boundary, so the first click a visitor makes crosses a real threshold.
const DEFAULT_DEMO_SCORE = 22;

function formatRupees(n: number): string {
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

// Section entrance stagger, in mount order. The animation itself (and
// the opacity-0 start state) lives in index.css behind a
// prefers-reduced-motion: no-preference media query, so a reduced-motion
// visitor gets every section at its final state immediately.
const STAGGER_MS = [0, 60, 120, 180, 220];

function staggerStyle(index: number): { animationDelay: string } {
  return { animationDelay: `${STAGGER_MS[index] ?? 220}ms` };
}

function ImpactBand({ headline, basis }: { headline: number; basis: string | null }) {
  const animated = useAnimatedNumber(headline);

  return (
    <div className={`${surface} p-6 flex flex-col lg:flex-row lg:items-center gap-6`}>
      <div className="lg:pr-6 lg:border-r border-app-rule shrink-0">
        <p className={`${typeScale.caption} uppercase tracking-wide`}>
          Estimated impact, extrapolated
        </p>
        <p className="font-mono text-4xl font-bold text-app-ink mt-1 tabular-nums">
          {formatRupees(animated)}
          <span className="text-lg font-normal text-app-faint"> / month</span>
        </p>
      </div>
      {basis && <p className={`${typeScale.caption} leading-relaxed`}>{basis}</p>}
    </div>
  );
}

export default function Overview() {
  const [cost, setCost] = useState<CostAnalysis | null>(null);
  const [entityCount, setEntityCount] = useState<number | null>(null);
  const [demoScore, setDemoScore] = useState(DEFAULT_DEMO_SCORE);

  useEffect(() => {
    api
      .costAnalysis(5000, 150)
      .then(setCost)
      .catch(() => {
        // Non-critical for this tab — the Model Validation tab's Cost
        // Analysis section surfaces the real error if something's wrong.
      });
    api
      .listEntities()
      .then((d) => setEntityCount(d.entities.length))
      .catch(() => {});
  }, []);

  const thresholds = cost?.decision_thresholds ?? null;

  return (
    <div className="space-y-8">
      <section
        className="animate-card-in grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-6"
        style={staggerStyle(0)}
      >
        <div className="space-y-6">
          <div>
            <h2 className={typeScale.sectionTitle}>What this is</h2>
            <p className={`${typeScale.body} mt-2 max-w-xl`}>
              A transaction risk-scoring pipeline built for Razorpay's Buildathon (Track 2): a
              gradient-boosted model scores each transaction, deterministic rules turn that score plus
              an entity's recent history into an ALLOW/REVIEW/BLOCK decision, and a Gemini-backed agent
              generates a plain-language explanation alongside it — without gating the actual decision.
            </p>
          </div>

          <div className={`${surface} p-5`}>
            <h3 className={typeScale.subTitle}>Try the decision rule</h3>
            {thresholds ? (
              <>
                <div className="mt-4">
                  <RiskGauge
                    score={demoScore}
                    reviewThreshold={thresholds.review}
                    blockThreshold={thresholds.block}
                    interactive={{ onPick: setDemoScore }}
                    label="Risk score"
                  />
                </div>
                <p className={`${typeScale.caption} mt-4`}>
                  This is the real REVIEW/BLOCK boundary the system uses — the same two numbers the
                  API decides with, not an illustration. Nothing is scored here; picking a value
                  just shows which side of the boundary it falls on.
                </p>
              </>
            ) : (
              <p className={`${typeScale.caption} mt-2`}>Loading the live decision boundary…</p>
            )}
          </div>
        </div>

        <div className={`${surface} p-5 h-fit`}>
          <h3 className={typeScale.subTitle}>At a glance</h3>
          <dl className="mt-2 divide-y divide-app-rule">
            <div className={groupedRow}>
              <dt className={groupedRowLabel}>Decision thresholds</dt>
              <dd className={groupedRowValue}>
                {thresholds ? `${thresholds.review} / ${thresholds.block}` : "—"}
              </dd>
            </div>
            <div className={groupedRow}>
              <dt className={groupedRowLabel}>Escalation cutoffs</dt>
              <dd className={groupedRowValue}>
                {cost?.escalation_cutoffs
                  ? `${cost.escalation_cutoffs.watch} / ${cost.escalation_cutoffs.elevated}`
                  : "—"}
              </dd>
            </div>
            <div className={groupedRow}>
              <dt className={groupedRowLabel}>Scoring mode</dt>
              <dd className={groupedRowValue}>
                {entityCount === null ? "—" : `${entityCount} replayable + custom`}
              </dd>
            </div>
            <div className={groupedRow}>
              <dt className={groupedRowLabel}>Model quality (ROC-AUC)</dt>
              <dd className={groupedRowValue}>{cost?.roc_auc ?? "—"}</dd>
            </div>
          </dl>
          <p className={`${typeScale.caption} mt-3`}>
            Every number here is read from the running API, not written into the page.
          </p>
        </div>
      </section>

      {cost?.headline_monthly_savings_estimate != null && (
        <section className="animate-card-in" style={staggerStyle(1)}>
          <ImpactBand
            headline={cost.headline_monthly_savings_estimate}
            basis={cost.headline_basis}
          />
        </section>
      )}

      <section className="animate-card-in" style={staggerStyle(2)}>
        <div className={`${noticeClass("neutral")} text-sm`}>
          This runs entirely on the historical, public <strong>IEEE-CIS Fraud Detection</strong>{" "}
          dataset from Kaggle — not live Razorpay data or a production Razorpay system. It's a
          demonstration built against Razorpay's own publicly stated engineering problems, not an
          integration with Razorpay's infrastructure.
        </div>
      </section>

      <section className="animate-card-in" style={staggerStyle(3)}>
        <h3 className={typeScale.subTitle}>What each tab actually answers</h3>
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {TAB_QUESTIONS.map((t) => (
            <div key={t.label} className={`${surface} p-4`}>
              <p className={`${typeScale.caption} font-medium text-app-ink uppercase tracking-wide`}>
                {t.label}
              </p>
              <p className={`${typeScale.body} mt-1.5`}>{t.question}</p>
            </div>
          ))}
        </div>

        <div className={`${surface} p-5 mt-4`}>
          <p className={`${typeScale.caption} font-medium text-app-ink uppercase tracking-wide`}>
            Model Validation
          </p>
          <p className={`${typeScale.body} mt-1.5 max-w-3xl`}>
            Every offline check this project ran on itself — each one a real script in this repo,
            run against the real test set.
          </p>
          <div className="flex flex-wrap gap-2 mt-3">
            {VALIDATION_CHIPS.map((chip) => (
              <span
                key={chip}
                className="text-xs text-app-muted border border-app-rule rounded-full px-3 py-1"
              >
                {chip}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="animate-card-in" style={staggerStyle(4)}>
        <h3 className={typeScale.subTitle}>Grounding</h3>
        <p className={`${typeScale.body} mt-2 max-w-3xl`}>
          The escalation logic and the consistency section (under Model Validation) are both
          grounded in problems Razorpay has described in its own engineering blog, not
          hypothetical ones:{" "}
          <a
            href="https://engineering.razorpay.com/meet-bumblebee-the-multi-agent-ai-architecture-that-changed-fraud-detection-at-razorpay-c2b6d5704f51"
            target="_blank"
            rel="noreferrer"
            className="text-app-accent-soft underline decoration-app-accent/40 underline-offset-2 hover:text-app-accent"
          >
            Meet Bumblebee
          </a>{" "}
          and{" "}
          <a
            href="https://engineering.razorpay.com/our-obsession-with-merchant-experience-breaking-the-risk-review-black-box-7fa38d699ef1"
            target="_blank"
            rel="noreferrer"
            className="text-app-accent-soft underline decoration-app-accent/40 underline-offset-2 hover:text-app-accent"
          >
            Breaking the Risk Review Black Box
          </a>
          .
        </p>
      </section>
    </div>
  );
}

import { useEffect, useState } from "react";
import { api } from "../api/client";
import { surface, typeScale } from "../theme";

const TAB_QUESTIONS: { label: string; question: string }[] = [
  { label: "Live Scoring", question: "Given one transaction and an entity's recent history, what would the model + LLM reviewer decide right now?" },
  { label: "Review Queue", question: "When a human disposes flagged verdicts as fraud or false-positive, how precise is the model — overall, and specifically when it escalated?" },
  { label: "Model Validation", question: "Every offline check this project ran on itself: cost-optimal threshold and sensitivity, does entity escalation actually help, does the model stay good over time, does the cold-start graph signal help brand-new entities, and does the system agree with itself on repeat looks at the same case?" },
];

function formatRupees(n: number): string {
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

export default function Overview() {
  const [headline, setHeadline] = useState<number | null>(null);
  const [headlineBasis, setHeadlineBasis] = useState<string | null>(null);

  useEffect(() => {
    api
      .costAnalysis(5000, 150)
      .then((data) => {
        setHeadline(data.headline_monthly_savings_estimate);
        setHeadlineBasis(data.headline_basis);
      })
      .catch(() => {
        // Non-critical for this tab — the Model Validation tab's Cost
        // Analysis section surfaces the real error if something's wrong.
      });
  }, []);

  return (
    <div className="space-y-8">
      <div className="max-w-3xl">
        <h2 className={typeScale.sectionTitle}>What this is</h2>
        <p className={`${typeScale.body} mt-2`}>
          A transaction risk-scoring pipeline built for Razorpay's Buildathon (Track 2): a
          gradient-boosted model scores each transaction, deterministic rules turn that score plus
          an entity's recent history into an ALLOW/REVIEW/BLOCK decision, and a Gemini-backed agent
          generates a plain-language explanation alongside it — without gating the actual decision.
        </p>
      </div>

      {headline !== null && (
        <div className={`${surface} p-5 max-w-3xl`}>
          <p className={`${typeScale.caption} uppercase tracking-wide`}>
            Estimated impact, extrapolated
          </p>
          <p className="text-4xl font-bold text-app-ink mt-1 tabular-nums">
            {formatRupees(headline)}
            <span className="text-lg font-normal text-app-faint"> / month</span>
          </p>
          {headlineBasis && <p className={`${typeScale.caption} mt-2 max-w-2xl`}>{headlineBasis}</p>}
        </div>
      )}

      <div className="max-w-3xl border border-amber-500/30 bg-amber-500/10 rounded-lg px-4 py-3">
        <p className="text-sm text-amber-200">
          This runs entirely on the historical, public <strong>IEEE-CIS Fraud Detection</strong>{" "}
          dataset from Kaggle — not live Razorpay data or a production Razorpay system. It's a
          demonstration built against Razorpay's own publicly stated engineering problems, not an
          integration with Razorpay's infrastructure.
        </p>
      </div>

      <div>
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
      </div>

      <div className="max-w-3xl">
        <h3 className={typeScale.subTitle}>Grounding</h3>
        <p className={`${typeScale.body} mt-2`}>
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
      </div>
    </div>
  );
}

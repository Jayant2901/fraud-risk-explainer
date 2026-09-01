import { surface, typeScale } from "../theme";

const TAB_QUESTIONS: { label: string; question: string }[] = [
  { label: "Live Scoring", question: "Given one transaction and an entity's recent history, what would the model + LLM reviewer decide right now?" },
  { label: "Review Queue", question: "When a human disposes flagged verdicts as fraud or false-positive, how precise is the model — overall, and specifically when it escalated?" },
  { label: "Cost-Optimal Threshold", question: "Given real costs for a missed fraud vs. a wrongly-blocked transaction, what score threshold actually minimizes money lost — and does the model stay good over time?" },
  { label: "Consistency", question: "Does this system agree with itself — across repeated LLM calls on the same case, and against its own deterministic rules?" },
];

export default function Overview() {
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
              <p className={`${typeScale.caption} font-medium text-neutral-200 uppercase tracking-wide`}>
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
          The escalation logic and the consistency tab are both grounded in problems Razorpay has
          described in its own engineering blog, not hypothetical ones:{" "}
          <a
            href="https://engineering.razorpay.com/meet-bumblebee-the-multi-agent-ai-architecture-that-changed-fraud-detection-at-razorpay-c2b6d5704f51"
            target="_blank"
            rel="noreferrer"
            className="text-indigo-300 underline decoration-indigo-500/40 underline-offset-2 hover:text-indigo-200"
          >
            Meet Bumblebee
          </a>{" "}
          and{" "}
          <a
            href="https://engineering.razorpay.com/our-obsession-with-merchant-experience-breaking-the-risk-review-black-box-7fa38d699ef1"
            target="_blank"
            rel="noreferrer"
            className="text-indigo-300 underline decoration-indigo-500/40 underline-offset-2 hover:text-indigo-200"
          >
            Breaking the Risk Review Black Box
          </a>
          .
        </p>
      </div>
    </div>
  );
}

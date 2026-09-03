import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import LiveScoring from "./LiveScoring";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    listEntities: vi.fn(),
    listTransactions: vi.fn(),
    getEscalation: vi.fn(),
    resetEntity: vi.fn(),
    score: vi.fn(),
    scoreCustom: vi.fn(),
    getExplanation: vi.fn(),
    costAnalysis: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

// The gauge's score arrival and the explanation typewriter are motion.
// This suite asserts on settled content, so it runs as a reduced-motion
// visitor — jsdom has no matchMedia at all, so this both provides it and
// pins the preference.
function mockReducedMotion() {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: query.includes("prefers-reduced-motion: reduce"),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    onchange: null,
    dispatchEvent: () => false,
  }));
}

function mockHappyPathApis() {
  mockReducedMotion();
  mockedApi.costAnalysis.mockResolvedValue({
    eval_report: null,
    defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
    params: { fraud_loss: 5000, fp_cost: 150 },
    headline_monthly_savings_estimate: null,
    headline_basis: null,
    cost_curve: [],
    decision_thresholds: { review: 34, block: 71 },
    escalation_cutoffs: { watch: 0.8, elevated: 3.6 },
    roc_auc: 0.9541,
  });
  mockedApi.listEntities.mockResolvedValue({ entities: ["entity-a", "entity-b"] });
  mockedApi.listTransactions.mockResolvedValue({
    entity_id: "entity-a",
    count: 1,
    transactions: [{ index: 0, TransactionAmt: 100, TransactionDT: 1000, ProductCD: "W" }],
  });
  mockedApi.getEscalation.mockResolvedValue({
    state: "NORMAL",
    recent_verdict_count: 0,
    recent_risky_count: 0,
    recent_verdicts: [],
  });
}

describe("LiveScoring", () => {
  it("renders the entity dropdown populated from api.listEntities", async () => {
    mockHappyPathApis();

    render(<LiveScoring />);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "entity-a" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "entity-b" })).toBeInTheDocument();
    });
  });

  it("calls listEntities on mount", async () => {
    mockHappyPathApis();

    render(<LiveScoring />);

    await waitFor(() => expect(mockedApi.listEntities).toHaveBeenCalled());
  });

  it("shows the empty-state prompt before anything has been scored", async () => {
    mockHappyPathApis();

    render(<LiveScoring />);

    expect(await screen.findByRole("button", { name: /Score this transaction/i })).toBeInTheDocument();
    expect(screen.getByText(/Pick an entity and transaction number/i)).toBeInTheDocument();
  });

  it("fetches transactions and escalation for the first entity once entities load", async () => {
    mockHappyPathApis();

    render(<LiveScoring />);

    await waitFor(() => {
      expect(mockedApi.listTransactions).toHaveBeenCalledWith("entity-a");
      expect(mockedApi.getEscalation).toHaveBeenCalledWith("entity-a");
    });
  });

  it("surfaces an error message if listEntities rejects", async () => {
    mockHappyPathApis();
    mockedApi.listEntities.mockRejectedValue(new Error("network down"));

    render(<LiveScoring />);

    expect(await screen.findByText(/network down/i)).toBeInTheDocument();
  });

  describe("custom transaction mode", () => {
    it("shows a custom-mode prompt and hides the entity dropdown once selected", async () => {
      mockHappyPathApis();

      render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });

      fireEvent.click(screen.getByRole("tab", { name: /Score custom/i }));

      expect(screen.queryByLabelText(/Entity \(card\/account fingerprint\)/i)).not.toBeInTheDocument();
      expect(screen.getByLabelText(/Transaction amount/i)).toBeInTheDocument();
    });

    it("submits the custom form and renders the returned results", async () => {
      mockHappyPathApis();
      mockedApi.scoreCustom.mockResolvedValue({
        risk_score: 12.3,
        above_threshold: false,
        top_factors: [],
        escalation_before: {
          state: "NORMAL",
          recent_verdict_count: 0,
          recent_risky_count: 0,
          recent_verdicts: [],
        },
        decision: { action: "ALLOW", escalated_due_to_history: false },
        verdict_id: "v-custom-1",
      });

      render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });

      fireEvent.click(screen.getByRole("tab", { name: /Score custom/i }));
      fireEvent.click(screen.getByRole("button", { name: /Score this transaction/i }));

      await waitFor(() => expect(mockedApi.scoreCustom).toHaveBeenCalled());
      const payload = mockedApi.scoreCustom.mock.calls.at(-1)![0];
      expect(payload.TransactionAmt).toBe(100);
      expect(payload.attach_to_entity_id).toBeUndefined();

      expect(await screen.findByText("12.3")).toBeInTheDocument();
      // Twice: the gauge's band for this score, and the final automated
      // decision. They agree here — the gauge reads the same thresholds
      // the backend decided with.
      expect(screen.getAllByText("ALLOW")).toHaveLength(2);
    });

    it("passes attach_to_entity_id when an entity is selected in the attach dropdown", async () => {
      mockHappyPathApis();
      mockedApi.scoreCustom.mockResolvedValue({
        risk_score: 55.0,
        above_threshold: true,
        top_factors: [],
        escalation_before: {
          state: "WATCH",
          recent_verdict_count: 3,
          recent_risky_count: 2,
          recent_verdicts: ["ALLOW", "REVIEW", "REVIEW"],
        },
        decision: { action: "REVIEW", escalated_due_to_history: false },
        verdict_id: "v-custom-2",
      });
      mockedApi.getEscalation.mockResolvedValue({
        state: "WATCH",
        recent_verdict_count: 3,
        recent_risky_count: 2,
        recent_verdicts: ["ALLOW", "REVIEW", "REVIEW"],
      });

      render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });

      fireEvent.click(screen.getByRole("tab", { name: /Score custom/i }));
      fireEvent.change(screen.getByLabelText(/Attach to entity/i), { target: { value: "entity-b" } });
      fireEvent.click(screen.getByRole("button", { name: /Score this transaction/i }));

      await waitFor(() => expect(mockedApi.scoreCustom).toHaveBeenCalled());
      const payload = mockedApi.scoreCustom.mock.calls.at(-1)![0];
      expect(payload.attach_to_entity_id).toBe("entity-b");
    });
  });

  describe("replay-at-speed (Play control)", () => {
    // Mocks in this file aren't cleared between it() blocks (no global
    // clearMocks), so without this reset, mockedApi.score's call count
    // would carry over from earlier tests in this describe block.
    beforeEach(() => {
      vi.clearAllMocks();
    });

    function mockThreeTxnEntity() {
      mockedApi.listEntities.mockResolvedValue({ entities: ["entity-a", "entity-b"] });
      mockedApi.listTransactions.mockResolvedValue({
        entity_id: "entity-a",
        count: 3,
        transactions: [
          { index: 0, TransactionAmt: 100, TransactionDT: 1000, ProductCD: "W" },
          { index: 1, TransactionAmt: 200, TransactionDT: 2000, ProductCD: "W" },
          { index: 2, TransactionAmt: 300, TransactionDT: 3000, ProductCD: "W" },
        ],
      });
      mockedApi.getEscalation.mockResolvedValue({
        state: "NORMAL",
        recent_verdict_count: 0,
        recent_risky_count: 0,
        recent_verdicts: [],
      });
      mockedApi.score.mockResolvedValue({
        risk_score: 10,
        above_threshold: false,
        top_factors: [],
        escalation_before: { state: "NORMAL", recent_verdict_count: 0, recent_risky_count: 0, recent_verdicts: [] },
        decision: { action: "ALLOW", escalated_due_to_history: false },
        verdict_id: "v-play-1",
      });
    }

    afterEach(() => {
      vi.useRealTimers();
    });

    it("steps through the sequence automatically at the expected cadence and stops at the end", async () => {
      mockThreeTxnEntity();
      render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });
      await waitFor(() => expect(mockedApi.listTransactions).toHaveBeenCalledWith("entity-a"));

      vi.useFakeTimers();
      fireEvent.click(screen.getByRole("button", { name: "Play" }));

      await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
      expect(mockedApi.score).toHaveBeenNthCalledWith(1, "entity-a", 0);

      await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
      expect(mockedApi.score).toHaveBeenNthCalledWith(2, "entity-a", 1);

      await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
      expect(mockedApi.score).toHaveBeenNthCalledWith(3, "entity-a", 2);
      expect(mockedApi.score).toHaveBeenCalledTimes(3);

      // No further ticks should be scheduled once the sequence is exhausted.
      await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
      expect(mockedApi.score).toHaveBeenCalledTimes(3);
      expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    });

    it("clears the pending tick when the component unmounts mid-play", async () => {
      mockThreeTxnEntity();
      const { unmount } = render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });
      await waitFor(() => expect(mockedApi.listTransactions).toHaveBeenCalledWith("entity-a"));

      vi.useFakeTimers();
      fireEvent.click(screen.getByRole("button", { name: "Play" }));

      unmount();

      await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
      expect(mockedApi.score).not.toHaveBeenCalled();
    });

    it("stops playback and clears the pending tick when the user switches entities mid-play", async () => {
      mockThreeTxnEntity();
      render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });
      await waitFor(() => expect(mockedApi.listTransactions).toHaveBeenCalledWith("entity-a"));

      vi.useFakeTimers();
      fireEvent.click(screen.getByRole("button", { name: "Play" }));

      fireEvent.change(screen.getByLabelText(/Entity \(card\/account fingerprint\)/i), {
        target: { value: "entity-b" },
      });

      await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
      expect(mockedApi.score).not.toHaveBeenCalled();
      expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    });
  });
  describe("result gauge and explanation reveal", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    function mockScoredResult(overrides: Record<string, unknown> = {}) {
      mockHappyPathApis();
      mockedApi.score.mockResolvedValue({
        risk_score: 80,
        above_threshold: true,
        top_factors: [],
        escalation_before: {
          state: "ELEVATED",
          recent_verdict_count: 3,
          recent_risky_count: 3,
          recent_verdicts: ["BLOCK", "BLOCK", "REVIEW"],
        },
        decision: { action: "BLOCK", escalated_due_to_history: true },
        verdict_id: "v-1",
        ...overrides,
      });
      mockedApi.getExplanation.mockResolvedValue({ status: "pending" });
    }

    async function scoreOnce() {
      render(<LiveScoring />);
      await screen.findByRole("option", { name: "entity-a" });
      // The button stays disabled until this entity's transactions land.
      const scoreButton = screen.getByRole("button", { name: /Score this transaction/i });
      await waitFor(() => expect(scoreButton).toBeEnabled());
      fireEvent.click(scoreButton);
      await waitFor(() => expect(mockedApi.score).toHaveBeenCalled());
    }

    it("renders the score in a gauge showing both real thresholds", async () => {
      mockScoredResult();

      await scoreOnce();

      expect(await screen.findByText("80")).toBeInTheDocument();
      // Tick marks come from the fetched decision_thresholds, not constants.
      expect(screen.getByText("34")).toBeInTheDocument();
      expect(screen.getByText("71")).toBeInTheDocument();
    });

    it("fetches the decision thresholds once, not per score", async () => {
      mockScoredResult();

      await scoreOnce();
      fireEvent.click(screen.getByRole("button", { name: /Score this transaction/i }));
      await waitFor(() => expect(mockedApi.score).toHaveBeenCalledTimes(2));

      expect(mockedApi.costAnalysis).toHaveBeenCalledTimes(1);
    });

    it("calls out an escalated decision in words, not just color", async () => {
      mockScoredResult();

      await scoreOnce();

      expect(
        await screen.findByText(/Escalated due to this entity's recent history/i)
      ).toBeInTheDocument();
    });

    it("says nothing about escalation when the score alone drove the decision", async () => {
      mockScoredResult({ decision: { action: "BLOCK", escalated_due_to_history: false } });

      await scoreOnce();

      expect(await screen.findByText("80")).toBeInTheDocument();
      expect(screen.queryByText(/Escalated due to this entity's recent history/i)).not.toBeInTheDocument();
    });

    it("still renders the score if the thresholds request fails", async () => {
      mockScoredResult();
      mockedApi.costAnalysis.mockRejectedValue(new Error("no cost analysis"));

      await scoreOnce();

      expect(await screen.findByText("80")).toBeInTheDocument();
      expect(screen.queryByText("34")).not.toBeInTheDocument();
    });

    it("reveals the full explanation and rationale once they arrive", async () => {
      mockScoredResult();
      mockedApi.getExplanation.mockResolvedValue({
        status: "ready",
        verdict: {
          explanation: "This card has three prior blocked attempts today.",
          action: "BLOCK",
          escalated_due_to_history: true,
          rationale: "Repeat velocity from one fingerprint.",
        },
      });

      await scoreOnce();

      // Under reduced motion both land complete and untruncated.
      expect(
        await screen.findByText("This card has three prior blocked attempts today.")
      ).toBeInTheDocument();
      expect(
        await screen.findByText(/Repeat velocity from one fingerprint\./)
      ).toBeInTheDocument();
    });

    it("keeps the pending state until the explanation is ready", async () => {
      mockScoredResult();

      await scoreOnce();

      expect(await screen.findByText(/Generating explanation/i)).toBeInTheDocument();
    });
  });
});

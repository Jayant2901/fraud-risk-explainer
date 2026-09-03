import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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

function mockHappyPathApis() {
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
      expect(screen.getByText("ALLOW")).toBeInTheDocument();
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
});

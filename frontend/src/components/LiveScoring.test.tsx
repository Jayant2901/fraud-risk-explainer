import { render, screen, waitFor } from "@testing-library/react";
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
});

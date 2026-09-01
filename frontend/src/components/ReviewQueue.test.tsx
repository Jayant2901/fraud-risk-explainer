import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ReviewQueue from "./ReviewQueue";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    listReviewQueue: vi.fn(),
    disposeReviewItem: vi.fn(),
    reviewQueueMetrics: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

const SAMPLE_ITEM = {
  verdict_id: "v1",
  entity_id: "entity-a",
  txn_index: 2,
  risk_score: 55.0,
  decision: { action: "REVIEW" as const, escalated_due_to_history: false },
  baseline_decision: { action: "REVIEW" as const, escalated_due_to_history: false },
  escalated_due_to_history: false,
  disposition: null,
  disposed_at: null,
};

const EMPTY_METRICS = {
  total_disposed: 0,
  overall_precision: null,
  escalated_count: 0,
  escalated_precision: null,
  non_escalated_count: 0,
  non_escalated_precision: null,
};

describe("ReviewQueue", () => {
  it("renders pending items with entity, score, and action", async () => {
    mockedApi.listReviewQueue.mockResolvedValue({ items: [SAMPLE_ITEM] });
    mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);

    render(<ReviewQueue />);

    expect(await screen.findByText("entity-a")).toBeInTheDocument();
    expect(screen.getByText(/Risk score: 55/)).toBeInTheDocument();
    expect(screen.getByText("REVIEW")).toBeInTheDocument();
  });

  it("shows the empty state when nothing is pending", async () => {
    mockedApi.listReviewQueue.mockResolvedValue({ items: [] });
    mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);

    render(<ReviewQueue />);

    expect(await screen.findByText(/Nothing pending/i)).toBeInTheDocument();
  });

  it("flags escalation-triggered items with the baseline action", async () => {
    mockedApi.listReviewQueue.mockResolvedValue({
      items: [
        {
          ...SAMPLE_ITEM,
          decision: { action: "BLOCK" as const, escalated_due_to_history: true },
          baseline_decision: { action: "REVIEW" as const, escalated_due_to_history: false },
          escalated_due_to_history: true,
        },
      ],
    });
    mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);

    render(<ReviewQueue />);

    expect(await screen.findByText(/escalated \(baseline: REVIEW\)/)).toBeInTheDocument();
  });

  it("confirming fraud calls the API and refreshes the list", async () => {
    mockedApi.listReviewQueue
      .mockResolvedValueOnce({ items: [SAMPLE_ITEM] })
      .mockResolvedValueOnce({ items: [] });
    mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);
    mockedApi.disposeReviewItem.mockResolvedValue({ ...SAMPLE_ITEM, disposition: "CONFIRMED_FRAUD" });

    render(<ReviewQueue />);

    const confirmBtn = await screen.findByRole("button", { name: /Confirm Fraud/i });
    fireEvent.click(confirmBtn);

    await waitFor(() => expect(mockedApi.disposeReviewItem).toHaveBeenCalledWith("v1", "CONFIRMED_FRAUD"));
    await waitFor(() => expect(screen.getByText(/Nothing pending/i)).toBeInTheDocument());
  });

  it("marking a false positive calls the API with the right disposition", async () => {
    mockedApi.listReviewQueue.mockResolvedValue({ items: [SAMPLE_ITEM] });
    mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);
    mockedApi.disposeReviewItem.mockResolvedValue({ ...SAMPLE_ITEM, disposition: "FALSE_POSITIVE" });

    render(<ReviewQueue />);

    const btn = await screen.findByRole("button", { name: /Mark False Positive/i });
    fireEvent.click(btn);

    await waitFor(() => expect(mockedApi.disposeReviewItem).toHaveBeenCalledWith("v1", "FALSE_POSITIVE"));
  });

  it("renders the precision metrics panel when dispositions exist", async () => {
    mockedApi.listReviewQueue.mockResolvedValue({ items: [] });
    mockedApi.reviewQueueMetrics.mockResolvedValue({
      total_disposed: 4,
      overall_precision: 0.75,
      escalated_count: 2,
      escalated_precision: 0.5,
      non_escalated_count: 2,
      non_escalated_precision: 1.0,
    });

    render(<ReviewQueue />);

    expect(await screen.findByText("75.0%")).toBeInTheDocument();
    expect(screen.getByText("50.0%")).toBeInTheDocument();
    expect(screen.getByText("100.0%")).toBeInTheDocument();
  });

  it("surfaces an error message if the queue request rejects", async () => {
    mockedApi.listReviewQueue.mockRejectedValue(new Error("network down"));
    mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);

    render(<ReviewQueue />);

    expect(await screen.findByText(/network down/i)).toBeInTheDocument();
  });
});

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ReviewQueue from "./ReviewQueue";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    listReviewQueue: vi.fn(),
    disposeReviewItem: vi.fn(),
    reviewQueueMetrics: vi.fn(),
    addReviewNote: vi.fn(),
    relatedReviewItems: vi.fn(),
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
  created_at: "2024-01-01T00:00:00+00:00",
  notes: [],
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
  beforeEach(() => {
    mockedApi.relatedReviewItems.mockResolvedValue({ items: [] });
  });

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

  describe("notes", () => {
    it("shows an 'Add note' toggle when there are no notes yet, and submits a new note", async () => {
      mockedApi.listReviewQueue.mockResolvedValue({ items: [SAMPLE_ITEM] });
      mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);
      mockedApi.addReviewNote.mockResolvedValue({
        author: "Reviewer",
        text: "Looks suspicious",
        at: "2024-01-01T01:00:00+00:00",
      });

      render(<ReviewQueue />);

      const toggle = await screen.findByRole("button", { name: /Add note/i });
      fireEvent.click(toggle);

      const input = screen.getByPlaceholderText(/Add a note/i);
      fireEvent.change(input, { target: { value: "Looks suspicious" } });
      fireEvent.click(screen.getByRole("button", { name: /^Add$/i }));

      await waitFor(() =>
        expect(mockedApi.addReviewNote).toHaveBeenCalledWith("v1", "Looks suspicious")
      );
      expect(await screen.findByText("Looks suspicious")).toBeInTheDocument();
    });

    it("shows the note count when notes already exist", async () => {
      mockedApi.listReviewQueue.mockResolvedValue({
        items: [{ ...SAMPLE_ITEM, notes: [{ author: "Alice", text: "prior note", at: "2024-01-01T00:00:00+00:00" }] }],
      });
      mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);

      render(<ReviewQueue />);

      expect(await screen.findByRole("button", { name: /1 note/i })).toBeInTheDocument();
    });
  });

  describe("related items", () => {
    it("shows a count of other items for the same entity and expands to list them", async () => {
      mockedApi.listReviewQueue.mockResolvedValue({ items: [SAMPLE_ITEM] });
      mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);
      mockedApi.relatedReviewItems.mockResolvedValue({
        items: [
          {
            ...SAMPLE_ITEM,
            verdict_id: "v2",
            decision: { action: "BLOCK" as const, escalated_due_to_history: false },
            disposition: "CONFIRMED_FRAUD" as const,
          },
        ],
      });

      render(<ReviewQueue />);

      const toggle = await screen.findByRole("button", { name: /1 other item for this entity/i });
      fireEvent.click(toggle);

      expect(await screen.findByText("CONFIRMED_FRAUD")).toBeInTheDocument();
    });

    it("shows no related-items toggle when there are none", async () => {
      mockedApi.listReviewQueue.mockResolvedValue({ items: [SAMPLE_ITEM] });
      mockedApi.reviewQueueMetrics.mockResolvedValue(EMPTY_METRICS);
      mockedApi.relatedReviewItems.mockResolvedValue({ items: [] });

      render(<ReviewQueue />);

      await screen.findByText("entity-a");
      expect(screen.queryByText(/other item.*for this entity/i)).not.toBeInTheDocument();
    });
  });
});

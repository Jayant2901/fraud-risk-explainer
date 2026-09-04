import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Overview from "./Overview";
import { api, type CostAnalysis } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    costAnalysis: vi.fn(),
    listEntities: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

// The count-up and the entrance animations are motion; this suite asserts
// on final rendered values, so it runs as a reduced-motion visitor would.
// (window.matchMedia isn't implemented in jsdom at all — this both stubs
// it and pins the preference.)
beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal(
    "matchMedia",
    (query: string) => ({
      matches: query.includes("prefers-reduced-motion: reduce"),
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      onchange: null,
      dispatchEvent: () => false,
    })
  );
  mockedApi.listEntities.mockResolvedValue({ entities: ["e1", "e2", "e3"] });
});

const COST: CostAnalysis = {
  eval_report: null,
  defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
  params: { fraud_loss: 5000, fp_cost: 150 },
  headline_monthly_savings_estimate: 500000,
  headline_basis:
    "Extrapolated ... illustrative assumption for scale, not a real Razorpay volume figure.",
  cost_curve: [],
    decision_thresholds: { review: 34, block: 71 },
  escalation_cutoffs: { watch: 0.8, elevated: 3.6 },
  roc_auc: 0.9541,
};

describe("Overview impact band", () => {
  it("renders the headline savings number and its basis when available", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);

    render(<Overview />);

    expect(await screen.findByText(/₹5,00,000/)).toBeInTheDocument();
    expect(screen.getByText(/illustrative assumption for scale/)).toBeInTheDocument();
  });

  it("shows nothing extra when no headline is available yet", async () => {
    mockedApi.costAnalysis.mockResolvedValue({
      ...COST,
      headline_monthly_savings_estimate: null,
      headline_basis: null,
    });

    render(<Overview />);

    expect(await screen.findByText(/What this is/)).toBeInTheDocument();
    expect(screen.queryByText(/Estimated impact/)).not.toBeInTheDocument();
  });

  it("still renders the static content if the cost-analysis request rejects", async () => {
    mockedApi.costAnalysis.mockRejectedValue(new Error("network down"));

    render(<Overview />);

    expect(await screen.findByText(/What this is/)).toBeInTheDocument();
    expect(screen.queryByText(/Estimated impact/)).not.toBeInTheDocument();
  });
});

describe("Overview at-a-glance panel", () => {
  it("renders all four values from the API response", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);

    render(<Overview />);

    expect(await screen.findByText("34 / 71")).toBeInTheDocument();
    expect(screen.getByText("0.8 / 3.6")).toBeInTheDocument();
    expect(screen.getByText("3 replayable + custom")).toBeInTheDocument();
    expect(screen.getByText("0.9541")).toBeInTheDocument();
  });

  it("falls back to a dash rather than inventing numbers when the API fails", async () => {
    mockedApi.costAnalysis.mockRejectedValue(new Error("network down"));
    mockedApi.listEntities.mockRejectedValue(new Error("network down"));

    render(<Overview />);

    expect(await screen.findByText(/At a glance/)).toBeInTheDocument();
    // Decision thresholds, escalation cutoffs, ROC-AUC: still genuinely
    // unknown, so still a dash. Scoring mode is different — a failed
    // listEntities means "historical entities unavailable," a real,
    // explicit state (FIX-1), not the same "nothing loaded yet" dash.
    expect(screen.getAllByText("—")).toHaveLength(3);
    expect(screen.getByText(/Historical entities unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/custom scoring still works/)).toBeInTheDocument();
  });

  it("shows a dash for scoring mode while entities are still loading, not the unavailable message", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);
    mockedApi.listEntities.mockReturnValue(new Promise(() => {})); // never resolves

    render(<Overview />);

    await screen.findByText("34 / 71");
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText(/Historical entities unavailable/)).not.toBeInTheDocument();
  });

  it("renders a dash for ROC-AUC when the model hasn't recorded one", async () => {
    mockedApi.costAnalysis.mockResolvedValue({ ...COST, roc_auc: null });

    render(<Overview />);

    expect(await screen.findByText("34 / 71")).toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});

describe("Overview interactive decision-rule widget", () => {
  it("seeds the gauge below the review threshold, showing ALLOW", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);

    render(<Overview />);

    expect(await screen.findByText("ALLOW")).toBeInTheDocument();
  });

  it("updates the displayed action when a higher preset is clicked", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);

    render(<Overview />);
    fireEvent.click(await screen.findByRole("button", { name: "90" }));

    expect(screen.getByText("BLOCK")).toBeInTheDocument();
    expect(screen.queryByText("ALLOW")).not.toBeInTheDocument();
  });

  it("lands on REVIEW at exactly the review threshold", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);

    render(<Overview />);
    fireEvent.click(await screen.findByRole("button", { name: "34" }));

    expect(screen.getByText("REVIEW")).toBeInTheDocument();
  });

  it("does not score anything — the widget is purely local", async () => {
    mockedApi.costAnalysis.mockResolvedValue(COST);

    render(<Overview />);
    fireEvent.click(await screen.findByRole("button", { name: "90" }));

    expect(mockedApi.costAnalysis).toHaveBeenCalledTimes(1);
    expect("scoreCustom" in mockedApi).toBe(false);
  });
});

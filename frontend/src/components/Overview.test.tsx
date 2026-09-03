import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import Overview from "./Overview";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    costAnalysis: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

describe("Overview", () => {
  it("renders the headline savings number and its basis when available", async () => {
    mockedApi.costAnalysis.mockResolvedValue({
      eval_report: null,
      defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
      params: { fraud_loss: 5000, fp_cost: 150 },
      headline_monthly_savings_estimate: 500000,
      headline_basis: "Extrapolated ... illustrative assumption for scale, not a real Razorpay volume figure.",
    });

    render(<Overview />);

    expect(await screen.findByText(/₹5,00,000/)).toBeInTheDocument();
    expect(screen.getByText(/illustrative assumption for scale/)).toBeInTheDocument();
  });

  it("shows nothing extra when no headline is available yet", async () => {
    mockedApi.costAnalysis.mockResolvedValue({
      eval_report: null,
      defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
      params: { fraud_loss: 5000, fp_cost: 150 },
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

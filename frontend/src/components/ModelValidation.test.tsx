import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ModelValidation from "./ModelValidation";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    costAnalysis: vi.fn(),
    costSensitivity: vi.fn(),
    driftAnalysis: vi.fn(),
    escalationAblation: vi.fn(),
    coldStartAnalysis: vi.fn(),
    consistencyAnalysis: vi.fn(),
    shadowComparison: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

function mockAllReports() {
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
  mockedApi.costSensitivity.mockResolvedValue({ sensitivity: null, message: null });
  mockedApi.driftAnalysis.mockResolvedValue({ drift: null, message: null });
  mockedApi.escalationAblation.mockResolvedValue({ report: null, summary: null, message: "No ablation report yet" });
  mockedApi.coldStartAnalysis.mockResolvedValue({ report: null, message: "No cold-start report yet" });
  mockedApi.consistencyAnalysis.mockResolvedValue({ consistency: null, message: "No consistency report yet" });
  mockedApi.shadowComparison.mockResolvedValue({
    configured: false, total_scored: 0, agreement_rate: null, action_pairs: [],
    message: "No shadow model configured",
  });
}

describe("ModelValidation", () => {
  it("renders a header for every section", () => {
    mockAllReports();

    render(<ModelValidation />);

    expect(screen.getByRole("button", { name: /Cost-Optimal Threshold, Sensitivity & Drift/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Entity Escalation Ablation/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Cold-Start Graph Features/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Consistency/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Shadow Model Comparison/i })).toBeInTheDocument();
  });

  it("has the cost section open by default and fetches its data", async () => {
    mockAllReports();

    render(<ModelValidation />);

    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalled());
    expect(mockedApi.escalationAblation).not.toHaveBeenCalled();
  });

  it("expanding a closed section mounts it and fetches its data", async () => {
    mockAllReports();

    render(<ModelValidation />);

    fireEvent.click(screen.getByRole("button", { name: /Entity Escalation Ablation/i }));

    await waitFor(() => expect(mockedApi.escalationAblation).toHaveBeenCalled());
    expect(await screen.findByText(/No ablation report yet/)).toBeInTheDocument();
  });

  it("collapsing an open section hides its content", async () => {
    mockAllReports();

    render(<ModelValidation />);
    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalled());

    const costToggle = screen.getByRole("button", { name: /Cost-Optimal Threshold, Sensitivity & Drift/i });
    expect(screen.getByLabelText(/assumed avg\. fraud loss/i)).toBeInTheDocument();

    fireEvent.click(costToggle);

    expect(screen.queryByLabelText(/assumed avg\. fraud loss/i)).not.toBeInTheDocument();
  });
});

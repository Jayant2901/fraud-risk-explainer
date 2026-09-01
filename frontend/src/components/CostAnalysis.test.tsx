import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import CostAnalysis from "./CostAnalysis";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    costAnalysis: vi.fn(),
    costSensitivity: vi.fn(),
    driftAnalysis: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

function mockCostAnalysisResponse() {
  mockedApi.costAnalysis.mockResolvedValue({
    eval_report: null,
    defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
    params: { fraud_loss: 5000, fp_cost: 150 },
  });
  mockedApi.costSensitivity.mockResolvedValue({
    sensitivity: null,
    message: "No sensitivity sweep yet — run `python src/cost_sensitivity.py` to generate one.",
  });
  mockedApi.driftAnalysis.mockResolvedValue({
    drift: null,
    message: "No drift report yet — run `python src/drift_analysis.py` to generate one.",
  });
}

describe("CostAnalysis", () => {
  it("renders both numeric inputs with their default values", () => {
    mockCostAnalysisResponse();

    render(<CostAnalysis />);

    expect(screen.getByLabelText(/assumed avg\. fraud loss/i)).toHaveValue(5000);
    expect(screen.getByLabelText(/assumed cost per wrongly-flagged/i)).toHaveValue(150);
  });

  it("fetches cost analysis with the default values on mount", async () => {
    mockCostAnalysisResponse();

    render(<CostAnalysis />);

    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalledWith(5000, 150));
  });

  it("re-fetches cost analysis when the fraud-loss input changes", async () => {
    mockCostAnalysisResponse();

    render(<CostAnalysis />);
    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalledWith(5000, 150));

    fireEvent.change(screen.getByLabelText(/assumed avg\. fraud loss/i), { target: { value: "7000" } });

    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalledWith(7000, 150));
  });

  it("re-fetches cost analysis when the false-positive-cost input changes", async () => {
    mockCostAnalysisResponse();

    render(<CostAnalysis />);
    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalledWith(5000, 150));

    fireEvent.change(screen.getByLabelText(/assumed cost per wrongly-flagged/i), { target: { value: "300" } });

    await waitFor(() => expect(mockedApi.costAnalysis).toHaveBeenCalledWith(5000, 300));
  });

  it("shows the eval report when one comes back", async () => {
    mockedApi.costAnalysis.mockResolvedValue({
      eval_report: "ROC-AUC: 0.95",
      defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
      params: { fraud_loss: 5000, fp_cost: 150 },
    });
    mockedApi.costSensitivity.mockResolvedValue({ sensitivity: null, message: null });
    mockedApi.driftAnalysis.mockResolvedValue({ drift: null, message: null });

    render(<CostAnalysis />);

    expect(await screen.findByText(/ROC-AUC: 0.95/)).toBeInTheDocument();
  });

  it("prompts to train the model when no eval report exists yet", async () => {
    mockCostAnalysisResponse(); // eval_report: null

    render(<CostAnalysis />);

    expect(await screen.findByText(/train_model\.py/)).toBeInTheDocument();
  });

  it("surfaces an error message if the request rejects", async () => {
    mockedApi.costAnalysis.mockRejectedValue(new Error("server error"));
    mockedApi.costSensitivity.mockResolvedValue({ sensitivity: null, message: null });
    mockedApi.driftAnalysis.mockResolvedValue({ drift: null, message: null });

    render(<CostAnalysis />);

    expect(await screen.findByText(/server error/i)).toBeInTheDocument();
  });

  it("renders the sensitivity table when a sweep report exists", async () => {
    mockCostAnalysisResponse();
    mockedApi.costSensitivity.mockResolvedValue({
      sensitivity: {
        base_fraud_loss: 5000,
        base_fp_cost: 150,
        fraud_loss_multipliers: [1.0, 2.0],
        fp_cost_multipliers: [1.0],
        grid: [
          {
            fraud_loss_multiplier: 1.0,
            fp_cost_multiplier: 1.0,
            avg_fraud_loss: 5000,
            avg_fp_cost: 150,
            optimal_threshold: 0.33,
            optimal_total_cost: 4359900,
            estimated_savings_pct: 9.4,
          },
          {
            fraud_loss_multiplier: 2.0,
            fp_cost_multiplier: 1.0,
            avg_fraud_loss: 10000,
            avg_fp_cost: 150,
            optimal_threshold: 0.22,
            optimal_total_cost: 8000000,
            estimated_savings_pct: 27.9,
          },
        ],
      },
      message: null,
    });

    render(<CostAnalysis />);

    expect(await screen.findByText("0.33")).toBeInTheDocument();
    expect(screen.getByText("0.22")).toBeInTheDocument();
  });

  it("renders the drift chart with the AUC range when a report exists", async () => {
    mockCostAnalysisResponse();
    mockedApi.driftAnalysis.mockResolvedValue({
      drift: {
        span_seconds: 3618231,
        num_buckets: 3,
        edges: [0, 1206077, 2412154, 3618231],
        buckets: [
          { bucket: 0, n: 100, n_fraud: 5, roc_auc: 0.95, precision: 0.2, recall: 0.85 },
          { bucket: 1, n: 100, n_fraud: 5, roc_auc: 0.93, precision: 0.2, recall: 0.85 },
          { bucket: 2, n: 100, n_fraud: 5, roc_auc: 0.96, precision: 0.2, recall: 0.85 },
        ],
      },
      message: null,
    });

    render(<CostAnalysis />);

    expect(await screen.findByRole("img", { name: /ROC-AUC across time buckets/i })).toBeInTheDocument();
    expect(screen.getByText(/ranges from 0.9300 to 0.9600/)).toBeInTheDocument();
  });

  it("prompts to run the drift script when no report exists yet", async () => {
    mockCostAnalysisResponse();

    render(<CostAnalysis />);

    expect(await screen.findByText(/drift_analysis\.py/)).toBeInTheDocument();
  });
});

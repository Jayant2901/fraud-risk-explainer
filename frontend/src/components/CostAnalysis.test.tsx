import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import CostAnalysis from "./CostAnalysis";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    costAnalysis: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

function mockCostAnalysisResponse() {
  mockedApi.costAnalysis.mockResolvedValue({
    eval_report: null,
    defaults: { avg_fraud_loss: 5000, avg_fp_cost: 150 },
    params: { fraud_loss: 5000, fp_cost: 150 },
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

    render(<CostAnalysis />);

    expect(await screen.findByText(/server error/i)).toBeInTheDocument();
  });
});

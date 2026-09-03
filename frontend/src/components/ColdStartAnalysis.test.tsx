import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ColdStartAnalysis from "./ColdStartAnalysis";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    coldStartAnalysis: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

describe("ColdStartAnalysis", () => {
  it("prompts to run the script when no report exists yet", async () => {
    mockedApi.coldStartAnalysis.mockResolvedValue({
      report: null,
      message: "No cold-start report yet — run `python src/graph_features_ablation.py` to generate one.",
    });

    render(<ColdStartAnalysis />);

    expect(await screen.findByText(/graph_features_ablation\.py/)).toBeInTheDocument();
  });

  it("renders the report text when one exists", async () => {
    mockedApi.coldStartAnalysis.mockResolvedValue({
      report: "Cold-start ablation report\n===========================",
      message: null,
    });

    render(<ColdStartAnalysis />);

    expect(await screen.findByText(/Cold-start ablation report/)).toBeInTheDocument();
  });

  it("surfaces an error message if the request rejects", async () => {
    mockedApi.coldStartAnalysis.mockRejectedValue(new Error("server error"));

    render(<ColdStartAnalysis />);

    expect(await screen.findByText(/server error/i)).toBeInTheDocument();
  });
});

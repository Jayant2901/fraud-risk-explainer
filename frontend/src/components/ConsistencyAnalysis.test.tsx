import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ConsistencyAnalysis from "./ConsistencyAnalysis";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    consistencyAnalysis: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

describe("ConsistencyAnalysis", () => {
  it("prompts to run the script when no report exists yet", async () => {
    mockedApi.consistencyAnalysis.mockResolvedValue({
      consistency: null,
      message: "No consistency report yet — run `python src/consistency_analysis.py` to generate one.",
    });

    render(<ConsistencyAnalysis />);

    expect(await screen.findByText(/consistency_analysis\.py/)).toBeInTheDocument();
  });

  it("renders Part A boundary fragility numbers", async () => {
    mockedApi.consistencyAnalysis.mockResolvedValue({
      consistency: {
        part_a_boundary_fragility: { n_flagged: 1000, n_near_boundary: 250, fraction_near_boundary: 0.25 },
        part_b_pairs: [],
      },
      message: null,
    });

    render(<ConsistencyAnalysis />);

    expect(await screen.findByText(/250 of 1,000 flagged transactions/)).toBeInTheDocument();
    expect(screen.getByText(/25%/)).toBeInTheDocument();
  });

  it("renders Part B pair rows including insufficient-data status", async () => {
    mockedApi.consistencyAnalysis.mockResolvedValue({
      consistency: {
        part_a_boundary_fragility: { n_flagged: 10, n_near_boundary: 2, fraction_near_boundary: 0.2 },
        part_b_pairs: [
          {
            status: "ok",
            n_calls: 5,
            n_excluded_fallback: 1,
            n_valid: 4,
            modal_action: "BLOCK",
            self_consistency_rate: 1.0,
            cross_agreement: true,
            band: "clear_block",
            row_index: 12,
            risk_score: 95.0,
            escalation_context: "NORMAL",
            deterministic_action: "BLOCK",
          },
          {
            status: "insufficient_data",
            n_calls: 5,
            n_excluded_fallback: 4,
            n_valid: 1,
            modal_action: null,
            self_consistency_rate: null,
            cross_agreement: null,
            band: "near_40_boundary",
            row_index: 3,
            risk_score: 41.0,
            escalation_context: "ELEVATED",
            deterministic_action: "BLOCK",
          },
        ],
      },
      message: null,
    });

    render(<ConsistencyAnalysis />);

    expect(await screen.findByText("clear_block")).toBeInTheDocument();
    expect(screen.getByText("BLOCK")).toBeInTheDocument();
    expect(screen.getByText("insufficient data")).toBeInTheDocument();
  });

  it("surfaces an error message if the request rejects", async () => {
    mockedApi.consistencyAnalysis.mockRejectedValue(new Error("server error"));

    render(<ConsistencyAnalysis />);

    expect(await screen.findByText(/server error/i)).toBeInTheDocument();
  });
});

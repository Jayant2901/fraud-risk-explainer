import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ShadowScoring from "./ShadowScoring";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    shadowComparison: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

describe("ShadowScoring", () => {
  it("prompts to configure a shadow model when none is set", async () => {
    mockedApi.shadowComparison.mockResolvedValue({
      configured: false, total_scored: 0, agreement_rate: null, action_pairs: [],
      message: "No shadow model configured — set SHADOW_MODEL_PATH to compare a candidate model against live decisions before promoting it.",
    });

    render(<ShadowScoring />);

    expect(await screen.findByText(/SHADOW_MODEL_PATH/)).toBeInTheDocument();
  });

  it("prompts for data when configured but nothing has been scored yet", async () => {
    mockedApi.shadowComparison.mockResolvedValue({
      configured: true, total_scored: 0, agreement_rate: null, action_pairs: [], message: null,
    });

    render(<ShadowScoring />);

    expect(await screen.findByText(/hasn't scored anything yet/)).toBeInTheDocument();
  });

  it("renders the agreement rate and action-pair breakdown once data exists", async () => {
    mockedApi.shadowComparison.mockResolvedValue({
      configured: true,
      total_scored: 10,
      agreement_rate: 0.8,
      action_pairs: [
        { live_action: "REVIEW", shadow_action: "REVIEW", count: 8 },
        { live_action: "REVIEW", shadow_action: "ALLOW", count: 2 },
      ],
      message: null,
    });

    render(<ShadowScoring />);

    expect(await screen.findByText("80.0%")).toBeInTheDocument();
    expect(screen.getByText("REVIEW → REVIEW")).toBeInTheDocument();
    expect(screen.getByText("REVIEW → ALLOW")).toBeInTheDocument();
  });

  it("surfaces an error message if the request rejects", async () => {
    mockedApi.shadowComparison.mockRejectedValue(new Error("server error"));

    render(<ShadowScoring />);

    expect(await screen.findByText(/server error/i)).toBeInTheDocument();
  });
});

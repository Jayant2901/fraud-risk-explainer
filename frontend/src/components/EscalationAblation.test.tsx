import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import EscalationAblation from "./EscalationAblation";
import { api, type EscalationAblationSummary } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    escalationAblation: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

const SUMMARY: EscalationAblationSummary = {
  n_transactions: 118108,
  baseline: {
    n_fraud: 4064,
    n_legit: 114044,
    flagged_fraud: 3590,
    flagged_legit: 13312,
    recall: 0.8834,
    false_flag_rate: 0.1167,
  },
  adjusted: {
    n_fraud: 4064,
    n_legit: 114044,
    flagged_fraud: 3632,
    flagged_legit: 16776,
    recall: 0.8937,
    false_flag_rate: 0.1471,
  },
  flips: { n_flips: 5933, n_flips_fraud: 193, precision: 0.0325 },
  sweep: [
    { watch_threshold: 0.8, elevated_threshold: 2.0, recall: 0.9031, false_flag_rate: 0.1794, cost: 5039450 },
    { watch_threshold: 1.2, elevated_threshold: 2.0, recall: 0.9031, false_flag_rate: 0.1794, cost: 5039450 },
    { watch_threshold: 0.8, elevated_threshold: 2.8, recall: 0.8971, false_flag_rate: 0.1594, cost: 4817150 },
    { watch_threshold: 0.8, elevated_threshold: 3.6, recall: 0.8937, false_flag_rate: 0.1471, cost: 4676400 },
  ],
};

describe("EscalationAblation", () => {
  it("prompts to run the script when no report exists yet", async () => {
    mockedApi.escalationAblation.mockResolvedValue({
      report: null,
      summary: null,
      message: "No ablation report yet — run `python src/escalation_ablation.py` to generate one.",
    });

    render(<EscalationAblation />);

    expect(await screen.findByText(/escalation_ablation\.py/)).toBeInTheDocument();
  });

  it("keeps the full text report available when one exists", async () => {
    mockedApi.escalationAblation.mockResolvedValue({
      report: "Escalation ablation study\n==========================",
      summary: null,
      message: null,
    });

    render(<EscalationAblation />);

    expect(await screen.findByText(/Escalation ablation study/)).toBeInTheDocument();
  });

  it("surfaces an error message if the request rejects", async () => {
    mockedApi.escalationAblation.mockRejectedValue(new Error("server error"));

    render(<EscalationAblation />);

    expect(await screen.findByText(/server error/i)).toBeInTheDocument();
  });

  describe("charts", () => {
    it("charts baseline vs escalation-adjusted with the real rates", async () => {
      mockedApi.escalationAblation.mockResolvedValue({
        report: "text",
        summary: SUMMARY,
        message: null,
      });

      render(<EscalationAblation />);

      expect(
        await screen.findByRole("img", { name: /baseline versus escalation-adjusted/i })
      ).toBeInTheDocument();
      // Both recalls and both false-flag rates, from the summary, not the text.
      expect(screen.getByText("0.8834")).toBeInTheDocument();
      expect(screen.getByText("0.8937")).toBeInTheDocument();
      expect(screen.getByText("0.1167")).toBeInTheDocument();
      expect(screen.getByText("0.1471")).toBeInTheDocument();
    });

    it("reports the escalation-triggered flip precision", async () => {
      mockedApi.escalationAblation.mockResolvedValue({
        report: "text",
        summary: SUMMARY,
        message: null,
      });

      render(<EscalationAblation />);

      expect(await screen.findByText(/5,933/)).toBeInTheDocument();
      expect(screen.getByText(/193 were fraud/)).toBeInTheDocument();
    });

    it("charts the sweep once per distinct elevated cutoff, not per grid row", async () => {
      mockedApi.escalationAblation.mockResolvedValue({
        report: "text",
        summary: SUMMARY,
        message: null,
      });

      render(<EscalationAblation />);

      // 4 sweep rows but only 3 distinct elevated cutoffs — the duplicate
      // watch candidate must not become a second point.
      const chart = await screen.findByRole("img", {
        name: /candidate elevated pressure cutoffs/i,
      });
      const polyline = chart.querySelector("polyline");
      expect(polyline?.getAttribute("points")?.trim().split(/\s+/)).toHaveLength(3);
    });

    it("shows no sweep chart when the summary has only one cutoff", async () => {
      mockedApi.escalationAblation.mockResolvedValue({
        report: "text",
        summary: { ...SUMMARY, sweep: [SUMMARY.sweep[0]] },
        message: null,
      });

      render(<EscalationAblation />);

      expect(
        await screen.findByRole("img", { name: /baseline versus escalation-adjusted/i })
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("img", { name: /candidate elevated pressure cutoffs/i })
      ).not.toBeInTheDocument();
    });
  });
});

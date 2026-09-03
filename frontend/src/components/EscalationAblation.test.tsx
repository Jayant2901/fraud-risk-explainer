import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import EscalationAblation from "./EscalationAblation";
import { api } from "../api/client";

vi.mock("../api/client", () => ({
  api: {
    escalationAblation: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

describe("EscalationAblation", () => {
  it("prompts to run the script when no report exists yet", async () => {
    mockedApi.escalationAblation.mockResolvedValue({
      report: null,
      message: "No ablation report yet — run `python src/escalation_ablation.py` to generate one.",
    });

    render(<EscalationAblation />);

    expect(await screen.findByText(/escalation_ablation\.py/)).toBeInTheDocument();
  });

  it("renders the report text when one exists", async () => {
    mockedApi.escalationAblation.mockResolvedValue({
      report: "Escalation ablation study\n==========================",
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
});

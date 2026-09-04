import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import RiskGauge from "./RiskGauge";
import { presetScores } from "../hooks";

const REVIEW = 34;
const BLOCK = 71;

function renderGauge(score: number, interactive?: { onPick: (s: number) => void }) {
  return render(
    <RiskGauge
      score={score}
      reviewThreshold={REVIEW}
      blockThreshold={BLOCK}
      interactive={interactive}
    />
  );
}

describe("RiskGauge banding", () => {
  // The boundary itself belongs to the higher band, matching
  // decision_rules.decide_action's `>=` comparisons.
  it.each([
    [REVIEW - 1, "ALLOW"],
    [REVIEW, "REVIEW"],
    [REVIEW + 1, "REVIEW"],
    [BLOCK - 1, "REVIEW"],
    [BLOCK, "BLOCK"],
    [BLOCK + 1, "BLOCK"],
  ])("score %i renders %s", (score, expected) => {
    renderGauge(score);
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it("renders the score and both threshold ticks from real props", () => {
    renderGauge(50);

    expect(screen.getByText("50")).toBeInTheDocument();
    expect(screen.getByText(String(REVIEW))).toBeInTheDocument();
    expect(screen.getByText(String(BLOCK))).toBeInTheDocument();
  });
});

describe("RiskGauge interactive mode", () => {
  it("renders no preset chips when not interactive", () => {
    renderGauge(50);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("calls onPick with the preset's value", () => {
    const onPick = vi.fn();
    renderGauge(10, { onPick });

    fireEvent.click(screen.getByRole("button", { name: "90" }));

    expect(onPick).toHaveBeenCalledWith(90);
  });

  it("derives presets from the thresholds rather than hardcoding them", () => {
    expect(presetScores(REVIEW, BLOCK)).toEqual([10, 34, 53, 71, 90]);
    expect(presetScores(20, 60)).toEqual([10, 20, 40, 60, 90]);
  });

  it("deduplicates presets that collide with a threshold", () => {
    expect(presetScores(10, 90)).toEqual([10, 50, 90]);
  });
});

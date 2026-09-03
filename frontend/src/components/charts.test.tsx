import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BarChart, LineChart } from "./charts";

describe("LineChart", () => {
  const POINTS = [
    { x: 0.1, y: 500 },
    { x: 0.2, y: 200 },
    { x: 0.3, y: 900 },
  ];

  it("renders one polyline vertex per data point", () => {
    render(<LineChart points={POINTS} ariaLabel="test curve" />);

    const chart = screen.getByRole("img", { name: "test curve" });
    expect(chart.querySelector("polyline")?.getAttribute("points")?.trim().split(/\s+/)).toHaveLength(3);
  });

  it("marks the minimum, not the last or largest point", () => {
    render(<LineChart points={POINTS} ariaLabel="test curve" formatY={(v) => `min:${v}`} />);

    expect(screen.getByText("min:200")).toBeInTheDocument();
  });

  it("labels the x-axis range from the real data", () => {
    render(<LineChart points={POINTS} ariaLabel="test curve" />);

    expect(screen.getByText("0.1")).toBeInTheDocument();
    expect(screen.getByText("0.3")).toBeInTheDocument();
  });

  it("renders reference lines with their labels", () => {
    render(
      <LineChart
        points={POINTS}
        ariaLabel="test curve"
        referenceLines={[{ x: 0.2, label: "REVIEW", status: "warning" }]}
      />
    );

    expect(screen.getByText("REVIEW")).toBeInTheDocument();
  });

  it("renders nothing rather than a broken axis when there is no data", () => {
    const { container } = render(<LineChart points={[]} ariaLabel="empty" />);

    expect(container).toBeEmptyDOMElement();
  });

  it("does not divide by zero when every point shares one value", () => {
    render(
      <LineChart points={[{ x: 1, y: 5 }, { x: 2, y: 5 }]} ariaLabel="flat" />
    );

    const points = screen.getByRole("img", { name: "flat" }).querySelector("polyline")
      ?.getAttribute("points");
    expect(points).not.toContain("NaN");
  });
});

describe("BarChart", () => {
  const BARS = [
    { label: "Baseline", value: 0.25, status: "neutral" as const },
    { label: "Adjusted", value: 0.5, status: "success" as const },
  ];

  it("renders every bar's label and formatted value", () => {
    render(<BarChart bars={BARS} ariaLabel="comparison" />);

    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("0.2500")).toBeInTheDocument();
    expect(screen.getByText("0.5000")).toBeInTheDocument();
  });

  it("scales bar widths against an explicit max", () => {
    const { container } = render(<BarChart bars={BARS} max={1} ariaLabel="comparison" />);

    const widths = [...container.querySelectorAll<HTMLElement>("[style*='width']")].map(
      (el) => el.style.width
    );
    expect(widths).toEqual(["25%", "50%"]);
  });

  it("uses a custom value formatter when given one", () => {
    render(
      <BarChart bars={BARS} max={1} formatValue={(v) => `${v * 100}%`} ariaLabel="comparison" />
    );

    expect(screen.getByText("25%")).toBeInTheDocument();
  });
});

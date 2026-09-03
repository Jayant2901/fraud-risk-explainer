import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import SegmentedControl from "./SegmentedControl";

const SEGMENTS = [
  { id: "a", label: "Alpha" },
  { id: "b", label: "Beta" },
];

describe("SegmentedControl", () => {
  it("marks only the active segment as selected", () => {
    render(<SegmentedControl segments={SEGMENTS} activeId="b" onChange={() => {}} />);

    expect(screen.getByRole("tab", { name: "Alpha" })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("tab", { name: "Beta" })).toHaveAttribute("aria-selected", "true");
  });

  it("exposes a tablist containing every segment", () => {
    render(<SegmentedControl segments={SEGMENTS} activeId="a" onChange={() => {}} />);

    expect(screen.getByRole("tablist")).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(2);
  });

  it("calls onChange with the clicked segment id", () => {
    const onChange = vi.fn();
    render(<SegmentedControl segments={SEGMENTS} activeId="a" onChange={onChange} />);

    fireEvent.click(screen.getByRole("tab", { name: "Beta" }));

    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("still fires onChange when the active segment is re-clicked", () => {
    const onChange = vi.fn();
    render(<SegmentedControl segments={SEGMENTS} activeId="a" onChange={onChange} />);

    fireEvent.click(screen.getByRole("tab", { name: "Alpha" }));

    expect(onChange).toHaveBeenCalledWith("a");
  });
});

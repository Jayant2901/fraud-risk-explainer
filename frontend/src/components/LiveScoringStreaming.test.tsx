import { describe, expect, it } from "vitest";

import { partialExplanation } from "../hooks";

// The model streams a JSON object, so mid-flight the accumulated text is
// a partially-written object. This is what keeps the reader from seeing
// raw JSON while the explanation is still arriving.
describe("partialExplanation", () => {
  it("returns nothing before the explanation field has started", () => {
    expect(partialExplanation("")).toBe("");
    expect(partialExplanation("{")).toBe("");
    expect(partialExplanation('{"action": "BLOCK", ')).toBe("");
  });

  it("extracts the explanation while it is still being written", () => {
    expect(partialExplanation('{"explanation": "The card ha')).toBe("The card ha");
  });

  it("extracts the complete value once the field closes", () => {
    expect(
      partialExplanation('{"explanation": "Three prior blocks.", "action": "BLOCK"}')
    ).toBe("Three prior blocks.");
  });

  it("unescapes quotes rather than showing the escape characters", () => {
    expect(partialExplanation('{"explanation": "The \\"card\\" was flagged')).toBe(
      'The "card" was flagged'
    );
  });

  it("renders an escaped newline as a space rather than a literal backslash-n", () => {
    // String.raw so this is the two characters the model actually
    // streams, not an already-interpreted newline.
    expect(partialExplanation(String.raw`{"explanation": "Line one.\nLine two.`)).toBe(
      "Line one. Line two."
    );
  });

  it("tolerates whitespace variations in the streamed JSON", () => {
    expect(partialExplanation('{ "explanation"   :   "spaced out')).toBe("spaced out");
  });
});

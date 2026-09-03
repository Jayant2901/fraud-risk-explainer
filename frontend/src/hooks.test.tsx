import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAnimatedNumber, usePrefersReducedMotion, useTypewriter } from "./hooks";

function stubMotionPreference(reduce: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: reduce && query.includes("prefers-reduced-motion: reduce"),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    onchange: null,
    dispatchEvent: () => false,
  }));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("usePrefersReducedMotion", () => {
  it("reports the reduce preference when the media query matches", () => {
    stubMotionPreference(true);
    expect(renderHook(() => usePrefersReducedMotion()).result.current).toBe(true);
  });

  it("reports false when it doesn't", () => {
    stubMotionPreference(false);
    expect(renderHook(() => usePrefersReducedMotion()).result.current).toBe(false);
  });
});

describe("useTypewriter", () => {
  const TEXT = "The card has three prior blocked attempts today.";

  it("renders the complete text immediately under reduced motion", () => {
    stubMotionPreference(true);

    const { result } = renderHook(() => useTypewriter(TEXT));

    expect(result.current).toBe(TEXT);
  });

  describe("with motion enabled", () => {
    beforeEach(() => {
      stubMotionPreference(false);
      vi.useFakeTimers();
    });

    it("builds the text up rather than showing it all at once", () => {
      const { result } = renderHook(() => useTypewriter(TEXT));

      expect(result.current).toBe("");

      act(() => {
        vi.advanceTimersByTime(100);
      });

      expect(result.current.length).toBeGreaterThan(0);
      expect(result.current.length).toBeLessThan(TEXT.length);
      expect(TEXT.startsWith(result.current)).toBe(true);
    });

    it("lands on exactly the full text, with no dropped tail", () => {
      const { result } = renderHook(() => useTypewriter(TEXT));

      act(() => {
        vi.advanceTimersByTime(10_000);
      });

      expect(result.current).toBe(TEXT);
    });

    it("stays empty while disabled, then reveals once enabled", () => {
      const { result, rerender } = renderHook(
        ({ enabled }) => useTypewriter(TEXT, enabled),
        { initialProps: { enabled: false } }
      );

      // Disabled means "not yet my turn" for the sequential rationale —
      // it renders the full text so nothing is lost if it never runs.
      expect(result.current).toBe(TEXT);

      rerender({ enabled: true });
      act(() => {
        vi.advanceTimersByTime(10_000);
      });

      expect(result.current).toBe(TEXT);
    });
  });
});

describe("useAnimatedNumber", () => {
  it("returns the target immediately under reduced motion", () => {
    stubMotionPreference(true);

    const { result } = renderHook(() => useAnimatedNumber(1785865));

    expect(result.current).toBe(1785865);
  });

  it("eases from 0 up to the target and settles exactly on it", () => {
    stubMotionPreference(false);
    vi.useFakeTimers();

    const { result } = renderHook(() => useAnimatedNumber(100, 650));

    expect(result.current).toBe(0);

    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(result.current).toBeGreaterThan(0);
    expect(result.current).toBeLessThan(100);

    act(() => {
      vi.advanceTimersByTime(650);
    });
    expect(result.current).toBe(100);
  });

  it("animates from the previous value on a second target, not from zero", () => {
    stubMotionPreference(false);
    vi.useFakeTimers();

    const { result, rerender } = renderHook(({ target }) => useAnimatedNumber(target, 450), {
      initialProps: { target: 80 },
    });

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toBe(80);

    rerender({ target: 20 });
    act(() => {
      vi.advanceTimersByTime(100);
    });

    // Moving down from 80 — never a reset to 0 and a re-climb.
    expect(result.current).toBeLessThan(80);
    expect(result.current).toBeGreaterThan(20);
  });
});

// Motion helpers shared across tabs. Every JS-driven animation in the app
// goes through one of these, so "respect prefers-reduced-motion" is
// decided once here rather than re-derived per component. CSS-driven
// animations use the @media (prefers-reduced-motion: no-preference)
// wrapper in index.css instead — same policy, other mechanism.
//
// Also holds a couple of small pure helpers (presetScores below) that
// used to live inside the component files that use them: oxlint's
// react(only-export-components) rule flags a file that exports both a
// component and a plain value/function (it breaks Fast Refresh), and
// this is the module that pattern already pointed at.
import { useEffect, useRef, useState } from "react";

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(query.matches);
    // Safari <14 only has the deprecated addListener form.
    if (query.addEventListener) {
      query.addEventListener("change", onChange);
      return () => query.removeEventListener("change", onChange);
    }
    query.addListener(onChange);
    return () => query.removeListener(onChange);
  }, []);

  return reduced;
}

const STEPS = 24;

function easeOutCubic(progress: number): number {
  return 1 - Math.pow(1 - progress, 3);
}

// Eases from whatever was last displayed to `target` over durationMs.
// Animating from the previous value (not from 0) is what makes a second
// arrival read as a move rather than a restart — Live Scoring's gauge
// depends on that; Overview's impact band starts at 0 simply because
// that's its first value.
export function useAnimatedNumber(target: number, durationMs = 650): number {
  const reducedMotion = usePrefersReducedMotion();
  const [value, setValue] = useState(reducedMotion ? target : 0);
  const fromRef = useRef(0);

  useEffect(() => {
    if (reducedMotion) {
      // Not derivable at render time: `value` is a stateful position
      // that setInterval below mutates across many ticks while
      // animating, so it must persist across renders as real state, not
      // be recomputed from props. Snapping it to `target` here is this
      // effect synchronizing that state with an external system change
      // — the OS/browser's prefers-reduced-motion setting flipping —
      // exactly the case oxlint's own rule text carves out.
      fromRef.current = target;
      setValue(target);
      return;
    }

    const from = fromRef.current;
    const delta = target - from;
    if (delta === 0) {
      setValue(target);
      return;
    }

    let step = 0;
    const timer = setInterval(() => {
      step += 1;
      if (step >= STEPS) {
        clearInterval(timer);
        fromRef.current = target;
        setValue(target);
        return;
      }
      setValue(from + delta * easeOutCubic(step / STEPS));
    }, durationMs / STEPS);

    return () => clearInterval(timer);
  }, [target, durationMs, reducedMotion]);

  return value;
}

// The model streams a JSON object, so mid-flight text looks like
// `{"explanation": "The card ha`. This pulls out just the explanation
// value so far, rather than showing the reader raw JSON — used by
// LiveScoring's ExplanationView.
export function partialExplanation(raw: string): string {
  const match = raw.match(/"explanation"\s*:\s*"((?:[^"\\]|\\.)*)/);
  if (!match) return "";
  return match[1].replace(/\\"/g, '"').replace(/\\n/g, " ").replace(/\\\\/g, "\\");
}

// Preset scores for RiskGauge's interactive mode: the two real
// thresholds, their midpoint, and a clear-allow / clear-block anchor
// either side. Derived from the passed-in thresholds so this never
// drifts from the live decision boundary.
export function presetScores(reviewThreshold: number, blockThreshold: number): number[] {
  const presets = [
    10,
    reviewThreshold,
    Math.round((reviewThreshold + blockThreshold) / 2),
    blockThreshold,
    90,
  ];
  return [...new Set(presets)].sort((a, b) => a - b);
}

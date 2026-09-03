// Motion helpers shared across tabs. Every JS-driven animation in the app
// goes through one of these, so "respect prefers-reduced-motion" is
// decided once here rather than re-derived per component. CSS-driven
// animations use the @media (prefers-reduced-motion: no-preference)
// wrapper in index.css instead — same policy, other mechanism.
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

const TYPE_TICK_MS = 18;
const CHARS_PER_TICK = 3;

// Presentational reveal of text the backend already returned in full —
// not a token stream. Returns the full text immediately under reduced
// motion, and always lands on exactly `fullText` (no dropped tail).
export function useTypewriter(fullText: string, enabled = true): string {
  const reducedMotion = usePrefersReducedMotion();
  const instant = reducedMotion || !enabled;
  const [shown, setShown] = useState(instant ? fullText : "");

  useEffect(() => {
    if (instant) {
      setShown(fullText);
      return;
    }

    setShown("");
    let cursor = 0;
    const timer = setInterval(() => {
      cursor += CHARS_PER_TICK;
      if (cursor >= fullText.length) {
        clearInterval(timer);
        setShown(fullText);
        return;
      }
      setShown(fullText.slice(0, cursor));
    }, TYPE_TICK_MS);

    return () => clearInterval(timer);
  }, [fullText, instant]);

  return shown;
}

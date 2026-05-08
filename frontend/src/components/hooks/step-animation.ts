import { useCallback, useEffect, useRef, useState } from "react";

export function useStepAnimation(delayMs: number) {
  const [animationStep, setAnimationStep] = useState(0);
  const timeRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = useCallback(() => {
    if (timeRef.current) {
      clearInterval(timeRef.current);
      timeRef.current = null;
    }
  }, []);

  useEffect(() => () => clearTimer(), [clearTimer]);

  const startAnimation = useCallback(
    (totalSteps: number) => {
      clearTimer();
      setAnimationStep(0);

      let currentStep = 0;
      timeRef.current = setInterval(() => {
        if (currentStep < totalSteps - 1) {
          currentStep++;
          setAnimationStep(currentStep);
        }
      }, delayMs);
    },
    [clearTimer, delayMs],
  );

  const reset = useCallback(() => {
    clearTimer();
    setAnimationStep(0);
  }, [clearTimer]);

  return {
    animationStep,
    setAnimationStep,
    startAnimation,
    clearTimer,
    reset,
  };
}

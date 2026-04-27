import { StepBar } from "@/components/ui/step-bar";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { ProfileView } from "@/components/views/ProfileView";
import { ProgressView } from "@/components/views/ProgressView";
import { ResultsView } from "@/components/views/ResultsView";
import { UploadView } from "@/components/views/UploadView";
import { matchJobs, parseCV } from "@/lib/api";
import { DEMO_CV, DEMO_MATCHES } from "@/lib/demo";
import { VIEWS, type AppState } from "@/lib/types";
import { useCallback, useEffect, useRef, useState } from "react";

const INITIAL_STATE: AppState = {
  view: "upload",
  file: null,
  cvData: null,
  matches: [],
  error: null,
  topN: 10,
  minScore: 0,
  sortBy: "score",
  isDemo: false,
};

const PARSE_STEPS = 5;
const MATCH_STEPS = 5;
const STEP_DELAY_MS = 500;

export function App() {
  const [state, setState] = useState(INITIAL_STATE);
  const [animStep, setAnimStep] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  useEffect(() => () => clearTimer(), []);

  const runStepAnim = useCallback(
    (totalSteps: number) => {
      clearTimer();
      setAnimStep(0);
      let step = 0;
      timerRef.current = setInterval(() => {
        if (step < totalSteps - 1) {
          step++;
          setAnimStep(step);
        }
      }, STEP_DELAY_MS);
    },
    [clearTimer],
  );

  const handleDemo = useCallback(() => {
    setState((s) => ({
      ...s,
      isDemo: true,
      file: new File([], "demo.pdf", { type: "application/pdf" }),
      error: null,
    }));
  }, []);

  const handleParse = useCallback(async () => {
    const isDemo = state.isDemo;
    const file = state.file;

    setState((s) => ({
      ...s,
      view: "parse",
      error: null,
    }));
    runStepAnim(PARSE_STEPS);

    try {
      const cvData = isDemo
        ? await new Promise<typeof DEMO_CV>((r) => setTimeout(() => r(DEMO_CV), 3000))
        : await parseCV(file!);

      clearTimer();
      setAnimStep(PARSE_STEPS);
      setState((s) => ({ ...s, cvData, view: "profile", error: null }));
    } catch (err) {
      clearTimer();
      setState((s) => ({
        ...s,
        view: "upload",
        error: `Parse failed: ${(err as Error).message}. Make sure the FastAPI server is running.`,
      }));
    }
  }, [state.isDemo, state.file, runStepAnim]);

  const handleMatch = useCallback(async () => {
    const isDemo = state.isDemo;
    const cvData = state.cvData!;
    const topN = state.topN;

    setState((s) => ({ ...s, view: "match", error: null }));
    runStepAnim(MATCH_STEPS);

    try {
      const matches = isDemo
        ? await new Promise<typeof DEMO_MATCHES>((r) => setTimeout(() => r(DEMO_MATCHES), 3200))
        : await matchJobs(cvData, topN);
      clearTimer();
      setAnimStep(MATCH_STEPS);
      setState((s) => ({ ...s, matches, view: "results", error: null }));
    } catch (err) {
      clearTimer();
      setState((s) => ({
        ...s,
        view: "profile",
        error: `Matching failed: ${(err as Error).message}. Make sure the FastAPI server is running on port 8000.`,
      }));
    }
  }, [state.isDemo, state.cvData, state.topN, runStepAnim]);

  const handleBack = useCallback(() => {
    setState((s) => {
      if (s.view === "results") {
        return { ...s, view: "profile", error: null };
      } else if (s.view === "profile") {
        return { ...s, view: "upload", error: null };
      }
      return { ...s, view: "upload", cvData: null, error: null };
    });
  }, []);

  const handleRestart = useCallback(() => {
    clearTimer();
    setAnimStep(0);
    setState(INITIAL_STATE);
  }, []);

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="bg-background border-border sticky top-0 z-50 border-b backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-245 items-center justify-between gap-4 px-6">
          <div className="flex flex-wrap items-center gap-5">
            <div className="cursor-pointer font-serif text-2xl" onClick={handleRestart}>
              recruit<span className="text-primary font-serif">.</span>
            </div>
          </div>

          {state.view !== "upload" && <StepBar current={VIEWS.indexOf(state.view)} />}

          <ThemeToggle />
        </div>
      </nav>

      {/* Views */}
      <main className="mx-auto max-w-245 px-6 py-12">
        {state.view === "upload" && (
          <UploadView state={state} setState={setState} onParse={handleParse} onDemo={handleDemo} />
        )}
        {state.view === "parse" && <ProgressView type="parse" step={animStep} />}
        {state.view === "profile" && state.cvData && (
          <ProfileView
            cv={state.cvData}
            state={state}
            setState={setState}
            onMatch={handleMatch}
            onBack={handleBack}
          />
        )}
        {state.view === "match" && <ProgressView type="match" step={animStep} />}
        {state.view === "results" && (
          <ResultsView
            state={state}
            setState={setState}
            onBack={handleBack}
            onRestart={handleRestart}
          />
        )}
      </main>
    </div>
  );
}

export default App;

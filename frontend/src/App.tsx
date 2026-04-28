import { StepBar } from "@/components/ui/step-bar";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { ProfileView } from "@/components/views/ProfileView";
import { ProgressView } from "@/components/views/ProgressView";
import { ResultsView } from "@/components/views/ResultsView";
import { UploadView } from "@/components/views/UploadView";
import { matchJobs, parseCV } from "@/lib/api";
import { DEMO_CV, DEMO_MATCHES } from "@/lib/demo";
import { appReducer, INITIAL_STATE } from "@/lib/reducer";
import { VIEWS } from "@/lib/types";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";

const PARSE_STEPS = 5;
const MATCH_STEPS = 5;
const STEP_DELAY_MS = 500;

export default function App() {
  const [state, dispatch] = useReducer(appReducer, INITIAL_STATE);
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

  const handleDemo = () => {
    dispatch({ type: "SET_DEMO" });
  };

  const handleParse = useCallback(async () => {
    const isDemo = state.isDemo;
    const file = state.file;

    dispatch({ type: "START_PARSING" });
    runStepAnim(PARSE_STEPS);

    try {
      const cvData = isDemo
        ? await new Promise<typeof DEMO_CV>((r) => setTimeout(() => r(DEMO_CV), 3000))
        : await parseCV(file!);

      clearTimer();
      setAnimStep(PARSE_STEPS);
      dispatch({ type: "PARSING_SUCCESS", payload: cvData });
    } catch (err) {
      clearTimer();
      dispatch({
        type: "SET_UPLOAD_ERROR",
        payload: `Parse failed: ${(err as Error).message}. Make sure the FastAPI server is running on port 8000.`,
      });
    }
  }, [state.isDemo, state.file, runStepAnim]);

  const handleMatch = useCallback(async () => {
    const isDemo = state.isDemo;
    const cvData = state.cvData!;
    const topN = state.topN;

    dispatch({ type: "START_MATCH" });
    runStepAnim(MATCH_STEPS);

    try {
      const matches = isDemo
        ? await new Promise<typeof DEMO_MATCHES>((r) => setTimeout(() => r(DEMO_MATCHES), 3200))
        : await matchJobs(cvData, topN);
      clearTimer();
      setAnimStep(MATCH_STEPS);
      dispatch({ type: "MATCH_SUCCESS", payload: matches });
    } catch (err) {
      clearTimer();
      dispatch({
        type: "MATCH_ERROR",
        payload: `Matching failed: ${(err as Error).message}. Make sure the FastAPI server is running on port 8000.`,
      });
    }
  }, [state.isDemo, state.cvData, state.topN, runStepAnim]);

  const handleBack = () => {
    dispatch({ type: "GO_BACK" });
  };

  const handleRestart = () => {
    clearTimer();
    setAnimStep(0);
    dispatch({ type: "RESTART" });
  };

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
          <UploadView
            file={state.file}
            error={state.error}
            isDemo={state.isDemo}
            dispatch={dispatch}
            onParse={handleParse}
            onDemo={handleDemo}
          />
        )}
        {state.view === "parse" && <ProgressView type="parse" step={animStep} />}
        {state.view === "profile" && state.cvData && (
          <ProfileView
            cv={state.cvData}
            topN={state.topN}
            dispatch={dispatch}
            onMatch={handleMatch}
            onBack={handleBack}
          />
        )}
        {state.view === "match" && <ProgressView type="match" step={animStep} />}
        {state.view === "results" && (
          <ResultsView
            cvData={state.cvData}
            matches={state.matches}
            minScore={state.minScore}
            sortBy={state.sortBy}
            dispatch={dispatch}
            onBack={handleBack}
            onRestart={handleRestart}
          />
        )}
      </main>
    </div>
  );
}


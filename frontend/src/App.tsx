import { useStepAnimation } from "@/components/hooks/step-animation";
import { BackendStatus } from "@/components/ui/backend-status";
import { StepBar } from "@/components/ui/step-bar";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { ProfileView } from "@/components/views/ProfileView";
import { ProgressView } from "@/components/views/ProgressView";
import { ResultsView } from "@/components/views/ResultsView";
import { UploadView } from "@/components/views/UploadView";
import { matchJobs, parseCV } from "@/lib/api";
import { DEMO_CV, DEMO_MATCHES } from "@/lib/demo";
import { appReducer, INITIAL_STATE } from "@/lib/reducer";
import { VIEWS, type AppView } from "@/lib/types";
import React, { useCallback, useEffect, useReducer } from "react";

const PARSE_STEPS = 5;
const MATCH_STEPS = 5;
const STEP_DELAY_MS = 500;

export default function App() {
  const [state, dispatch] = useReducer(appReducer, INITIAL_STATE);
  const { animationStep, setAnimationStep, startAnimation, clearTimer, reset } =
    useStepAnimation(STEP_DELAY_MS);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [state.view]);

  const handleDemo = () => {
    dispatch({ type: "SET_DEMO" });
  };

  const handleParse = useCallback(async () => {
    const isDemo = state.isDemo;
    const file = state.file;

    dispatch({ type: "START_PARSING" });
    startAnimation(PARSE_STEPS);

    try {
      const cvData = isDemo
        ? await new Promise<typeof DEMO_CV>((r) => setTimeout(() => r(DEMO_CV), 3000))
        : await parseCV(file!);

      clearTimer();
      setAnimationStep(PARSE_STEPS);
      dispatch({ type: "PARSING_SUCCESS", payload: cvData });
    } catch (err) {
      clearTimer();
      dispatch({
        type: "SET_UPLOAD_ERROR",
        payload: `Parse failed: ${(err as Error).message}.`,
      });
    }
  }, [state.isDemo, state.file, startAnimation]);

  const handleMatch = useCallback(async () => {
    const isDemo = state.isDemo;
    const cvData = state.cvData!;
    const topN = state.topN;

    dispatch({ type: "START_MATCH" });
    startAnimation(MATCH_STEPS);

    try {
      const matches = isDemo
        ? await new Promise<typeof DEMO_MATCHES>((r) => setTimeout(() => r(DEMO_MATCHES), 3200))
        : await matchJobs(cvData, topN);
      clearTimer();
      startAnimation(MATCH_STEPS);
      dispatch({ type: "MATCH_SUCCESS", payload: matches });
    } catch (err) {
      clearTimer();
      dispatch({
        type: "MATCH_ERROR",
        payload: `Match failed: ${(err as Error).message}.`,
      });
    }
  }, [state.isDemo, state.cvData, state.topN, startAnimation]);

  const handleBack = () => {
    dispatch({ type: "GO_BACK" });
  };

  const handleRestart = () => {
    reset();
    dispatch({ type: "RESTART" });
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* Nav */}
      <NavBar view={state.view} onRestart={handleRestart} />

      {/* Views */}
      <main className="mx-auto w-full max-w-245 flex-1 px-6 py-12">
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
        {state.view === "parse" && <ProgressView type="parse" step={animationStep} />}
        {state.view === "profile" && state.cvData && (
          <ProfileView
            cv={state.cvData}
            topN={state.topN}
            error={state.error}
            dispatch={dispatch}
            onMatch={handleMatch}
            onBack={handleBack}
          />
        )}
        {state.view === "match" && <ProgressView type="match" step={animationStep} />}
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

      {state.view !== "parse" && state.view !== "match" && <Footer />}
    </div>
  );
}

function NavBar({ view, onRestart }: { view: AppView; onRestart: () => void }) {
  return (
    <nav className="bg-background border-border sticky top-0 z-50 border-b backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-245 items-center justify-between gap-4 px-6">
        <div className="flex flex-wrap items-center gap-5">
          <div className="cursor-pointer font-serif text-2xl" onClick={onRestart}>
            recruit<span className="text-primary font-serif">.</span>
          </div>
        </div>
        {view !== "upload" ? <StepBar current={VIEWS.indexOf(view)} /> : <BackendStatus />}
        <ThemeToggle />
      </div>
    </nav>
  );
}

const Footer = React.memo(function Footer() {
  return (
    <footer className="border-border bg-background border-t py-6">
      <div className="mx-auto flex max-w-245 flex-col justify-between gap-8 px-6 md:flex-row">
        <div className="flex max-w-sm flex-col gap-4">
          <div className="font-serif text-xl font-medium">
            recruit<span className="text-primary font-serif">.</span>
          </div>
          <p className="text-muted-foreground text-sm leading-relaxed">
            AI-powered recruitment and CV parsing platform to help you find the perfect job matches.
          </p>
        </div>
        <div>
          <h3 className="mb-4 font-semibold">Built by</h3>
          <div className="flex flex-col gap-3">
            <a
              href="https://github.com/AtifChy"
              target="_blank"
              rel="noreferrer"
              className="group flex w-fit items-center gap-3"
            >
              <img
                className="border-border bg-muted group-hover:border-primary inline-block h-8 w-8 rounded-full border-2 object-cover transition-transform group-hover:scale-110"
                src="https://github.com/AtifChy.png"
                alt="Md. Iftakhar Awal Chowdhury"
              />
              <span className="text-muted-foreground group-hover:text-foreground text-sm transition-colors">
                Iftakhar
              </span>
            </a>
            <a
              href="https://github.com/raiyen-zayed-rakin"
              target="_blank"
              rel="noreferrer"
              className="group flex w-fit items-center gap-3"
            >
              <img
                className="border-border bg-muted group-hover:border-primary inline-block h-8 w-8 rounded-full border-2 object-cover transition-transform group-hover:scale-110"
                src="https://github.com/raiyen-zayed-rakin.png"
                alt="Raiyen Zayed Rakin"
              />
              <span className="text-muted-foreground group-hover:text-foreground text-sm transition-colors">
                Raiyen
              </span>
            </a>
            <a
              href="https://github.com/abdulbarikyieash"
              target="_blank"
              rel="noreferrer"
              className="group flex w-fit items-center gap-3"
            >
              <img
                className="border-border bg-muted group-hover:border-primary inline-block h-8 w-8 rounded-full border-2 object-cover transition-transform group-hover:scale-110"
                src="https://github.com/abdulbarikyieash.png"
                alt="Abdul Bari Kyieash"
              />
              <span className="text-muted-foreground group-hover:text-foreground text-sm transition-colors">
                Abdul
              </span>
            </a>
          </div>
        </div>
      </div>
      <div className="border-border text-muted-foreground mx-auto mt-6 max-w-245 border-t px-6 pt-6 text-center text-sm md:flex md:justify-between md:text-left">
        <p>&copy; {new Date().getFullYear()} RecruitBD. All rights reserved.</p>
        <div className="mt-4 flex justify-center gap-6 md:mt-0">
          <a href="#" className="hover:text-foreground transition-colors">
            Twitter
          </a>
          <a href="#" className="hover:text-foreground transition-colors">
            LinkedIn
          </a>
          <a
            href="https://github.com/raiyen-zayed-rakin/RecruitBD-AI"
            className="hover:text-foreground transition-colors"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
});

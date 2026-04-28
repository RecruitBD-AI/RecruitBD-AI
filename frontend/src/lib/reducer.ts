import type { AppState, CVData, JobMatch } from "@/lib/types";

export const INITIAL_STATE: AppState = {
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

export type AppAction =
  // Upload and demo actions
  | { type: "SET_FILE"; payload: File | null }
  | { type: "SET_DEMO" }
  | { type: "SET_UPLOAD_ERROR"; payload: string }

  // Parse
  | { type: "START_PARSING" }
  | { type: "PARSING_SUCCESS"; payload: CVData }
  | { type: "PARSING_ERROR"; payload: string }

  // Match
  | { type: "START_MATCH" }
  | { type: "MATCH_SUCCESS"; payload: JobMatch[] }
  | { type: "MATCH_ERROR"; payload: string }

  // Navigation
  | { type: "GO_BACK" }
  | { type: "RESTART" }

  // Settings
  | { type: "SET_TOP_N"; payload: number }
  | { type: "SET_MIN_SCORE"; payload: number }
  | { type: "SET_SORT_BY"; payload: AppState["sortBy"] };

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    // Upload and demo actions
    case "SET_FILE":
      return { ...state, file: action.payload, error: null, isDemo: false };
    case "SET_DEMO":
      return {
        ...state,
        isDemo: true,
        file: new File([], "demo_cv.pdf", { type: "application/pdf" }),
        error: null,
      };
    case "SET_UPLOAD_ERROR":
      return { ...state, error: action.payload };

    // Parse
    case "START_PARSING":
      return { ...state, view: "parse", error: null };
    case "PARSING_SUCCESS":
      return { ...state, view: "profile", cvData: action.payload, error: null };
    case "PARSING_ERROR":
      return { ...state, view: "upload", error: action.payload };

    // Match
    case "START_MATCH":
      return { ...state, view: "match", error: null };
    case "MATCH_SUCCESS":
      return { ...state, view: "results", matches: action.payload, error: null };
    case "MATCH_ERROR":
      return { ...state, view: "profile", error: action.payload };

    // Navigation
    case "GO_BACK":
      if (state.view === "results") return { ...state, view: "profile", error: null };
      if (state.view === "profile") return { ...state, view: "upload", error: null };
      return { ...state, view: "upload", cvData: null, error: null };
    case "RESTART":
      return { ...INITIAL_STATE };

    // Settings
    case "SET_TOP_N":
      return { ...state, topN: action.payload };
    case "SET_MIN_SCORE":
      return { ...state, minScore: action.payload };
    case "SET_SORT_BY":
      return { ...state, sortBy: action.payload };

    default:
      return state;
  }
}

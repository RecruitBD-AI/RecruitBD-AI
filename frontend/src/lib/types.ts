// Define TypeScript interfaces for the CV data structure
export interface CVData {
  name: string;
  email: string;
  phone: string[];
  location: string;
  summary: string;
  skills: string[];
  experience: ExperienceEntry[];
  education: EducationEntry[];
  languages: string[];
  certifications: string[];
  references: string[];
}

export interface ExperienceEntry {
  title: string;
  company: string;
  startDate: string;
  endDate: string;
  description: string;
}

export interface EducationEntry {
  degree: string;
  institution: string;
  year: string;
}

// Define TypeScript interfaces for the job matching results
export interface JobMatch {
  job_id: string;
  job_title: string;
  company: string;
  location: string;
  salary_range: string;
  deadline: string;
  final_score: number;
  breakdown: ScoreBreakdown;
}

export interface ScoreBreakdown {
  skill_match: number;
  education_match: number;
  experience_match: number;
  semantic_match: number;
  title_match: number;
  skill_source: number;
  seniority_penalty: number;
}

// Define typescript interface for the application state
export interface AppState {
  view: AppView;
  file: File | null;
  cvData: CVData | null;
  matches: JobMatch[];
  error: string | null;
  topN: number;
  minScore: number;
  sortBy: "score" | "skill" | "exp" | "edu";
  isDemo: boolean;
}

export const VIEWS = ["upload", "parse", "profile", "match", "results"] as const;
export type AppView = (typeof VIEWS)[number];

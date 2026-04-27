import type { CVData, JobMatch } from "@/lib/types";

const API = import.meta.env.VITE_API_URL;

export async function parseCV(file: File): Promise<CVData> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API}/parse-cv`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
    throw new Error(err.message ?? `Server error: ${response.status}`);
  }

  return response.json();
}

export async function matchJobs(cv: CVData, topN: number): Promise<JobMatch[]> {
  const response = await fetch(`${API}/match-jobs?top=${topN}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cv),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
    throw new Error(err.message ?? `Server error: ${response.status}`);
  }

  return response.json();
}

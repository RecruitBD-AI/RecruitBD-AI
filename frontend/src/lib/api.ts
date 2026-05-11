import type { CVData, Health, JobMatch } from "@/lib/types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

export async function parseCV(file: File): Promise<CVData> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/parse-cv`, {
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
  const response = await fetch(`${API_URL}/match-jobs?top=${topN}`, {
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

export async function checkHealth(): Promise<Health> {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
    throw new Error(err.message ?? `Server error: ${response.status}`);
  }
  return response.json();
}

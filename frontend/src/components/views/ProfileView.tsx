import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { AppState, CVData } from "@/types";
import { ArrowLeftIcon, ArrowRightIcon, BriefcaseIcon, DotIcon, MinusIcon } from "lucide-react";
import type React from "react";

interface ProfileViewProps {
  cv: CVData;
  state: AppState;
  setState: React.Dispatch<React.SetStateAction<AppState>>;
  onMatch: () => void;
  onBack: () => void;
}

const CORE_SKILLS = new Set([
  "python",
  "javascript",
  "typescript",
  "java",
  "react",
  "node.js",
  "django",
  "fastapi",
  "flask",
  "sql",
  "mongodb",
  "docker",
  "kubernetes",
  "aws",
  "azure",
  "gcp",
  "machine learning",
  "deep learning",
  "nlp",
  "flutter",
  "dart",
  "kotlin",
  "swift",
  "spring boot",
  "tensorflow",
  "pytorch",
  "git",
  "rest api",
  "redis",
  "postgresql",
  "linux",
]);

export function ProfileView({ cv, state, setState, onMatch, onBack }: ProfileViewProps) {
  const initials = (cv.name || "CV")
    .split(" ")
    .map((s) => s[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return (
    <div className="animate-fade-in-up">
      {/* Header */}
      <div className="mb-7 flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-4">
          <div className="border-primary bg-primary/15 text-primary flex h-14 w-14 items-center justify-center rounded-full border-2 font-serif text-xl">
            {initials}
          </div>
          <div>
            <h2 className="mb-0.5 text-xl font-semibold">{cv.name}</h2>
            <div className="text-muted-foreground flex flex-row text-sm">
              {cv.email || ""}
              {cv.location && (
                <>
                  <DotIcon size={20} />
                  {cv.location}
                </>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={onBack}>
            <ArrowLeftIcon /> Back
          </Button>
          <Button size="sm" onClick={onMatch}>
            <BriefcaseIcon /> Find jobs
          </Button>
        </div>
      </div>

      {cv.summary !== "" && (
        <Card className="mb-4 gap-2 p-5">
          <div className="text-muted-foreground text-[11px] font-medium tracking-widest uppercase">
            Summary
          </div>
          <div className="text-sm font-medium">{cv.summary}</div>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-4">
        {/* Skills */}
        <Card className="p-5">
          <div className="text-muted-foreground flex flex-row text-[11px] font-medium tracking-widest uppercase">
            Skills <DotIcon size={16} /> {cv.skills.length}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {cv.skills.map((s, i) => {
              const isCore = CORE_SKILLS.has(s.toLowerCase());
              return (
                <Badge
                  key={i}
                  variant="outline"
                  className={cn("text-xs", isCore ? "bg-primary/10 text-primary" : "bg-border")}
                >
                  {s}
                </Badge>
              );
            })}
          </div>
        </Card>

        {/* Experience */}
        <Card className="p-5">
          <div className="text-muted-foreground flex flex-row text-[11px] font-medium tracking-widest uppercase">
            Experience <DotIcon size={16} /> {cv.experience.length}{" "}
            {cv.experience.length === 1 ? "role" : "roles"}
          </div>
          {cv.experience.length === 0 ? (
            <div className="text-muted-foreground text-sm">No experience found</div>
          ) : (
            cv.experience.map((e, i) => (
              <div
                key={i}
                className={cn("pb-3", i < cv.experience.length - 1 ? "border-border border-b" : "")}
              >
                <div className="text-sm font-medium">{e.title || <MinusIcon size={20} />}</div>
                <div className="text-muted-foreground flex items-center text-xs">
                  {e.company}
                  {e.startDate && (
                    <>
                      <DotIcon size={16} />
                      {e.startDate}
                      {e.endDate && (
                        <>
                          <MinusIcon size={14} />
                          {e.endDate}
                        </>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))
          )}
        </Card>

        {/* Education */}
        <Card className="p-5">
          <div className="text-muted-foreground flex flex-row text-[11px] font-medium tracking-widest uppercase">
            Education
          </div>
          {cv.education.map((e, i) => (
            <div
              key={i}
              className={cn(
                "flex items-start justify-between pb-3",
                i < cv.education.length - 1 ? "border-border border-b" : "",
              )}
            >
              <div>
                <div className="text-sm font-medium">{e.degree}</div>
                <div className="text-muted-foreground text-xs">{e.institution}</div>
              </div>
              <Badge variant="outline" className="shrink-0 text-xs">
                {e.year}
              </Badge>
            </div>
          ))}
        </Card>

        {/* Match Settings */}
        <Card className="p-5">
          <div className="text-muted-foreground flex flex-row text-[11px] font-medium tracking-widest uppercase">
            Match Settings
          </div>
          <div>
            <div className="mb-2 text-sm">Top results to return</div>
            <Input
              type="number"
              value={state.topN}
              onChange={(e) => setState((s) => ({ ...s, topN: parseInt(e.target.value) || 10 }))}
              className="mb-2 max-w-25"
            />
            <div className="text-muted-foreground text-xs leading-relaxed">
              Scores jobs across skill match, education fit, experience level, semantic similarity,
              title alignment, and seniority penalty.
            </div>
            <Separator className="my-4" />
            <Button onClick={onMatch} className="w-full">
              <BriefcaseIcon /> Run matching <ArrowRightIcon />
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}

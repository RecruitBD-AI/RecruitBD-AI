import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import type { AppAction } from "@/lib/reducer";
import type { CVData } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  BadgeAlertIcon,
  BriefcaseIcon,
  DotIcon,
  MinusIcon,
} from "lucide-react";
import type React from "react";

interface ProfileViewProps {
  cv: CVData;
  topN: number;
  error: string | null;
  dispatch: React.Dispatch<AppAction>;
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

export function ProfileView({ cv, topN, error, dispatch, onMatch, onBack }: ProfileViewProps) {
  const initials = (cv.name || "CV")
    .split(" ")
    .map((s) => s[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  const handleTopNChange = (topN: number) => {
    dispatch({ type: "SET_TOP_N", payload: topN });
  };

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

      {error && (
        <div className="text-destructive bg-destructive/10 mb-6 flex items-center justify-center gap-2 rounded-md border p-3 text-[15px]">
          <BadgeAlertIcon size={18} />
          {error}
        </div>
      )}

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
                <div className="text-sm font-medium capitalize">
                  {e.title || <MinusIcon size={20} />}
                </div>
                <div className="text-muted-foreground flex items-center text-xs">
                  {e.type === "project" && e.tech && e.tech.length > 0 ? (
                    <div className="mb-1 pt-1">
                      {e.tech.map((t, j) => (
                        <Badge key={j} variant="outline" className="bg-border mr-1 mb-1 text-xs">
                          {t}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <>
                      {e.company !== "" ? e.company : "N/A"}

                      {e.startDate && (
                        <>
                          <DotIcon size={16} />
                          {e.startDate}

                          {e.endDate && <span>&nbsp;– {e.endDate}</span>}
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
        <MatchSettings topN={topN} onTopNChange={handleTopNChange} onMatch={onMatch} />
      </div>
    </div>
  );
}

function MatchSettings({
  topN,
  onTopNChange,
  onMatch,
}: {
  topN: number;
  onTopNChange: (topN: number) => void;
  onMatch: () => void;
}) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value > 0) {
      onTopNChange(value);
    }
  };

  return (
    <Card className="p-5">
      <div className="text-muted-foreground flex flex-row text-[11px] font-medium tracking-widest uppercase">
        Match Settings
      </div>
      <div className="flex h-full flex-col">
        <div className="mb-2 text-sm">Top results to return</div>
        <Input
          type="number"
          value={topN}
          onChange={handleChange}
          onKeyDown={(e) => e.key === "Enter" && onMatch()}
          className="mb-2 max-w-25"
        />
        <div className="text-muted-foreground text-xs leading-relaxed">
          Scores jobs across skill match, education fit, experience level, semantic similarity,
          title alignment, and seniority penalty.
        </div>
        <div className="mt-auto">
          <Separator className="my-4" />
          <Button onClick={onMatch} className="w-full">
            <BriefcaseIcon /> Run matching <ArrowRightIcon />
          </Button>
        </div>
      </div>
    </Card>
  );
}

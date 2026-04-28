import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface ProgressViewProps {
  type: "parse" | "match";
  step: number;
}

const PARSE_STEPS = [
  "Extracting text from document",
  "Parsing skills and experience",
  "Analysing education history",
  "Structuring candidate profile",
  "Generating CV summary",
] as const;

const MATCH_STEPS = [
  "Loading job index",
  "Encoding candidate profile",
  "Computing semantic similarity",
  "Scoring all job listings",
  "Ranking top matches",
] as const;

export function ProgressView({ type, step }: ProgressViewProps) {
  const steps = type === "parse" ? PARSE_STEPS : MATCH_STEPS;
  const label = type === "parse" ? "Parsing your CV" : "Finding matches";
  const pct = Math.round(((step + 1) / steps.length) * 100);
  const currentLabel = steps[Math.min(step, steps.length - 1)];

  return (
    <div className="animate-fade-in-up mx-auto max-w-150">
      {/* Header */}
      <div className="mb-10 text-center">
        {/* Spinner */}
        <div className="border-primary mx-auto mb-5 h-14 w-14 animate-spin rounded-full border-2 border-t-transparent"></div>
        <h2 className="mb-1.5 font-serif text-[26px]">
          {label}
          <em className="text-primary">…</em>
        </h2>
        <p className="text-muted-foreground text-sm">This takes a few seconds</p>
      </div>

      <Card className="py-2">
        <CardContent className="p-8">
          {/* Progress Bar */}
          <div className="mb-6">
            <div className="text-muted-foreground mb-2 flex justify-between text-sm font-medium">
              <span>{currentLabel}</span>
              <span>{pct}%</span>
            </div>
            <Progress
              value={pct}
              className="[&>div]:bg-border [&>div]:h-1 [&>div]:transition-all [&>div]:duration-500"
            />
          </div>

          {/* Step List */}
          <div className="flex flex-col gap-2.5">
            {steps.map((s, i) => {
              const isDone = i < step;
              const isActive = i === step;

              return (
                <div
                  key={i}
                  className={cn(
                    "flex items-center gap-2.5 text-[13px] transition-all",
                    isActive ? "text-foreground" : "text-muted-foreground",
                    i > step ? "opacity-40" : "opacity-100",
                  )}
                >
                  {/* Dot */}
                  <div
                    className={cn(
                      "bg-border h-2 w-2 shrink-0 rounded-full",
                      isDone ? "bg-ring" : "",
                      isActive ? "bg-primary animate-pulse-dot" : "",
                    )}
                  />
                  <span>{s}</span>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

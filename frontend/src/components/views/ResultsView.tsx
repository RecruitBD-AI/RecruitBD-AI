import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { AppState, JobMatch } from "@/types";
import {
  ArrowLeftIcon,
  BriefcaseIcon,
  CalendarIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  DollarSignIcon,
  ExternalLinkIcon,
  FrownIcon,
  ListIcon,
  MapPinIcon,
  MinusIcon,
  RefreshCwIcon,
  SparklesIcon,
  TrendingUpIcon,
} from "lucide-react";
import type React from "react";
import { useState } from "react";

interface ResultsViewProps {
  state: AppState;
  setState: React.Dispatch<React.SetStateAction<AppState>>;
  onBack: () => void;
  onRestart: () => void;
}

const sortItems = [
  { label: "Match score", value: "score" },
  { label: "Skill match", value: "skill" },
  { label: "Experience", value: "exp" },
  { label: "Education", value: "edu" },
];

function sortMatches(matches: JobMatch[], sortBy: AppState["sortBy"]): JobMatch[] {
  const copy = [...matches];
  if (sortBy === "score") copy.sort((a, b) => b.final_score - a.final_score);
  else if (sortBy === "skill")
    copy.sort((a, b) => (b.breakdown?.skill_match ?? 0) - (a.breakdown?.skill_match ?? 0));
  else if (sortBy === "exp")
    copy.sort(
      (a, b) => (b.breakdown?.experience_match ?? 0) - (a.breakdown?.experience_match ?? 0),
    );
  else if (sortBy === "edu")
    copy.sort((a, b) => (b.breakdown?.education_match ?? 0) - (a.breakdown?.education_match ?? 0));
  return copy;
}

export function ResultsView({ state, setState, onBack, onRestart }: ResultsViewProps) {
  const { cvData, matches, minScore, sortBy } = state;
  const name = cvData?.name || "Candidate";

  const sorted = sortMatches(matches, sortBy);
  const filtered = sorted.filter((j) => j.final_score >= minScore);

  const avgScore = sorted.length
    ? Math.round(sorted.reduce((a, b) => a + b.final_score, 0) / sorted.length)
    : 0;
  const topScore = sorted.length ? Math.round(sorted[0].final_score) : 0;

  return (
    <div className="animate-fade-in-up">
      {/* Header */}
      <Header name={name} onBack={onBack} onRestart={onRestart} />

      {/* Stat Cards */}
      <StatCards matches={matches} topScore={topScore} avgScore={avgScore} filtered={filtered} />

      {/* Controls */}
      <Controls
        minScore={minScore}
        setState={setState}
        sortBy={sortBy}
        filtered={filtered}
        matches={matches}
      />

      {/* Job Cards */}
      {filtered.length === 0 ? (
        <Card className="flex flex-col items-center gap-2 p-12">
          <FrownIcon size={48} className="text-chart-4" />
          <div className="mt-4 text-[16px] font-medium">No results above {minScore}%</div>
          <div className="text-muted-foreground text-sm">
            Lower the minimum score to see more matches.
          </div>
        </Card>
      ) : (
        <div key={sortBy} className="flex flex-col gap-3.5">
          {filtered.map((job, idx) => (
            <JobCard key={job.job_id ?? idx} job={job} rank={idx + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function Header({
  name,
  onBack,
  onRestart,
}: {
  name: string;
  onBack: () => void;
  onRestart: () => void;
}) {
  return (
    <div className="mb-6 flex flex-wrap justify-between">
      <div>
        <div className="text-muted-foreground mb-1 text-sm">Result for</div>
        <h2 className="font-serif text-2xl leading-relaxed font-semibold tracking-wider">{name}</h2>
      </div>
      <div className="flex items-center gap-1">
        <Button variant="outline" size="sm" onClick={onBack}>
          <ArrowLeftIcon /> Back
        </Button>
        <Button variant="outline" size="sm" onClick={onRestart}>
          <RefreshCwIcon /> New CV
        </Button>
      </div>
    </div>
  );
}

function StatCards({
  matches,
  topScore,
  avgScore,
  filtered,
}: {
  matches: JobMatch[];
  topScore: number;
  avgScore: number;
  filtered: JobMatch[];
}) {
  const statItems = [
    {
      label: "Jobs scanned",
      val: matches.length.toLocaleString(),
      accent: false,
      icon: ListIcon,
    },
    { label: "Top match", val: `${topScore}%`, accent: true, icon: SparklesIcon },
    { label: "Average score", val: `${avgScore}%`, accent: false, icon: TrendingUpIcon },
    { label: "Showing", val: String(filtered.length), accent: false, icon: BriefcaseIcon },
  ];

  return (
    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
      {statItems.map(({ label, val, accent, icon: Icon }) => (
        <Card key={label} className="flex flex-row items-center justify-between gap-1 px-4 py-4">
          <div>
            <div className="text-muted-foreground mb-1.5 text-[11px] tracking-wider uppercase">
              {label}
            </div>
            <h2 className={cn("text-2xl font-semibold", accent ? "text-primary" : "")}>{val}</h2>
          </div>
          <Icon className="text-muted-foreground" />
        </Card>
      ))}
    </div>
  );
}

function Controls({
  minScore,
  setState,
  sortBy,
  filtered,
  matches,
}: {
  minScore: number;
  setState: React.Dispatch<React.SetStateAction<AppState>>;
  sortBy: string;
  filtered: JobMatch[];
  matches: JobMatch[];
}) {
  const handleMinScoreChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 0 && value <= 100) {
      setState((s) => ({ ...s, minScore: value }));
    }
  };

  const handleSortByChange = (value: string | null) => {
    if (!value) return;
    setState((s) => ({
      ...s,
      sortBy: value as AppState["sortBy"],
    }));
  };

  return (
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
      <div className="flex gap-2">
        <div>
          <div className="text-muted-foreground mb-1.5 text-[11px] tracking-wider uppercase">
            Min Score
          </div>
          <Input type="number" value={minScore} onChange={handleMinScoreChange} className="w-22" />
        </div>
        <div>
          <div className="text-muted-foreground mb-1.5 text-[11px] tracking-wider uppercase">
            Sort By
          </div>
          <Select items={sortItems} value={sortBy} onValueChange={handleSortByChange}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectLabel>Sort by</SelectLabel>
                {sortItems.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="text-muted-foreground mb-1.5 text-[11px] tracking-widest uppercase">
        {filtered.length} / {matches.length} results
      </div>
    </div>
  );
}

function JobCard({ job, rank }: { job: JobMatch; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const breakdown = job.breakdown;

  const score = Math.round(job.final_score);

  const handleCardClick = () => {
    setExpanded((prev) => !prev);
  };

  return (
    <Card
      className={cn(
        "hover:border-primary/20 hover:shadow-primary border border-transparent p-0 transition-all duration-200 hover:shadow-md/30",
        expanded ? "border-primary/20 shadow-primary shadow-md/30 transition-all" : "",
      )}
      onClick={handleCardClick}
    >
      <CardContent className="p-0">
        {/* Header Row */}
        <div className="flex items-start gap-4 px-6 py-5">
          <div className="text-muted-foreground font-mono text-sm">#{rank}</div>
          <div className="min-w-0 flex-1">
            <div className="mb-1 pr-20 text-[16px] leading-snug font-medium">
              {job.job_title || "Untitled role"}
              {rank === 1 && (
                <Badge variant="outline" className="bg-primary/15 text-primary ml-2">
                  <SparklesIcon />
                  Top Match
                </Badge>
              )}
            </div>
            <div className="text-muted-foreground mb-3 text-[13px]">
              {job.company || "Company not listed"}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {job.location && (
                <Badge variant="outline" className="bg-chart-2/15 text-chart-2">
                  <MapPinIcon />
                  {job.location}
                </Badge>
              )}
              {job.salary_range && (
                <Badge variant="outline" className="bg-chart-3/15 text-chart-3">
                  <DollarSignIcon />
                  {job.salary_range}
                </Badge>
              )}
              {job.deadline && (
                <Badge variant="outline" className="bg-chart-4/15 text-chart-4">
                  <CalendarIcon />
                  {job.deadline}
                </Badge>
              )}
            </div>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1.5">
            <div
              className={cn(
                "text-xl font-semibold",
                score >= 75 ? "text-primary" : score >= 50 ? "text-chart-4" : "text-destructive",
              )}
            >
              {score}
              <span className="text-muted-foreground text-sm">%</span>
            </div>
            <div className="text-muted-foreground">
              {expanded ? <ChevronUpIcon size="18" /> : <ChevronDownIcon size="18" />}
            </div>
          </div>
        </div>

        {expanded && (
          <div className="border-t px-6 py-5">
            <div className="text-muted-foreground mb-3.5 text-[11px] font-medium tracking-wider uppercase">
              Score breakdown
            </div>
            <div className="mb-4 grid grid-cols-2 gap-3.5 sm:grid-cols-3">
              <BreakdownBar label="Skill" value={breakdown.skill_match} />
              <BreakdownBar label="Experience" value={breakdown.experience_match} />
              <BreakdownBar label="Education" value={breakdown.education_match} />
              <BreakdownBar label="Semantic" value={breakdown.semantic_match} />
              <BreakdownBar label="Title Fit" value={breakdown.title_match} />
              <BreakdownBar label="Skill Source" value={breakdown.skill_source} />
            </div>

            {breakdown.seniority_penalty !== undefined && (
              <div className="text-muted-foreground mb-3 text-[11px]">
                Seniority penalty: ×{breakdown.seniority_penalty}
              </div>
            )}

            <Separator className="mb-3.5" />

            <div className="flex items-center gap-2">
              {job.job_id && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    window.open(`https://jobs.bdjobs.com/jobdetails?id=${job.job_id}`, "_blank")
                  }
                >
                  View on BDJobs
                  <ExternalLinkIcon />
                </Button>
              )}
              <span className="text-muted-foreground ml-auto flex items-center gap-1 text-[11px]">
                ID: {job.job_id || <MinusIcon size={14} />}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function BreakdownBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      {/* Label Row */}
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-muted-foreground text-[11px] tracking-wider uppercase">{label}</span>
        <span className="text-xs font-medium">{Math.round(value)}%</span>
      </div>

      {/* Bar */}
      <div className="bg-border h-1 overflow-hidden rounded">
        <div
          className={cn(
            "bg-primary h-full rounded transition-all duration-500 ease-out",
            value >= 70 ? "bg-primary" : value >= 50 ? "bg-chart-4/80" : "bg-destructive",
          )}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import type { AppState } from "@/types";
import { ArrowRightIcon, FileTextIcon, UploadIcon, XIcon } from "lucide-react";
import type React from "react";
import { useCallback, useRef, useState } from "react";

interface UploadViewProps {
  state: AppState;
  setState: React.Dispatch<React.SetStateAction<AppState>>;
  onParse: () => void;
  onDemo: () => void;
}

type ProcessStep = {
  title: string;
  desc: string;
  color: string;
};

const PROCESS_STEPS: ProcessStep[] = [
  {
    title: "Parse",
    desc: "Extracts skills, experience, education and employment history using Ollama/gemma3.",
    color: "text-chart-1",
  },
  {
    title: "Embed",
    desc: "Encodes your CV summary with Sentence-BERT and computes cosine similarity across all jobs.",
    color: "text-chart-2",
  },
  {
    title: "Score",
    desc: "Weighted scoring across 6 dimensions: skill match, education, experience, semantics, title & seniority.",
    color: "text-chart-3",
  },
  {
    title: "Rank",
    desc: "Returns top N matches with full score breakdowns, salary, deadlines and apply links.",
    color: "text-chart-4",
  },
];

function StepCard({ item, index }: { item: ProcessStep; index: number }) {
  return (
    <Card className="flex flex-row items-start gap-3.5 px-4 py-4.5">
      <div className={`min-w-6 pt-0.5 text-xl font-medium ${item.color}`}>
        {String(index + 1).padStart(2, "0")}
      </div>
      <div>
        <div className="mb-0.5 text-sm font-semibold">{item.title}</div>
        <div className="text-muted-foreground text-xs leading-relaxed">{item.desc}</div>
      </div>
    </Card>
  );
}

export function UploadView({ state, setState, onParse, onDemo }: UploadViewProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!["pdf", "docx"].includes(ext ?? "")) {
        setState((s) => ({ ...s, error: "Only PDF and DOCX files are supported." }));
        return;
      }
      setState((s) => ({ ...s, file, error: null, isDemo: false }));
    },
    [setState],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const canParse = !!(state.file || state.isDemo);

  return (
    <div className="animate-fade-in-up">
      {/* Hero */}
      <div className="mb-14">
        <div className="mb-4 flex items-center gap-2.5">
          <Badge variant="outline" className="bg-primary/15 text-primary tracking-wider">
            New
          </Badge>
          <span className="text-muted-foreground text-[13px]">
            Powered by Ollama + Sentence-BERT
          </span>
        </div>
        <h1 className="mb-3.5 font-serif text-[clamp(36px,6vw,58px)] leading-[1.1]">
          Match your CV to
          <br />
          <em className="text-primary">the right jobs.</em>
        </h1>
        <p className="text-muted-foreground max-w-120 text-[15px]">
          Upload your CV and our engine parses your profile, then scores every available job against
          your skills, experience, and education.
        </p>
      </div>

      <div className="grid grid-cols-2 items-start gap-5">
        {/* Upload Card */}
        <Card className="gap-1 overflow-hidden py-2">
          <CardHeader className="border-border border-b pt-4">
            <p className="mb-0.5 text-sm font-bold">Upload your CV</p>
            <span className="text-muted-foreground text-xs">PDF or DOCX, max 10MB</span>
            {/* <Separator /> */}
          </CardHeader>

          <CardContent className="px-6 py-4">
            {state.error && (
              <div className="text-destructive bg-destructive/10 mb-4 rounded-md border px-2.5 py-3">
                {state.error}
              </div>
            )}

            {/* Drop Zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`hover:border-primary hover:bg-primary/5 mb-4 rounded-md border border-dashed px-9 py-6 text-center transition-all ${
                dragging ? "border-primary bg-primary/5" : "border-border bg-muted cursor-pointer"
              } `}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".pdf,.docx"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              <div className="border-border text-primary bg-background mx-auto mb-3.5 flex h-10 w-10 items-center justify-center rounded-sm border">
                <UploadIcon />
              </div>
              <div className="mb-1 text-sm font-medium">
                {state.file ? "File ready" : "Drop your CV here"}
              </div>
              <div className="text-muted-foreground text-xs">
                {state.file ? state.file.name : "or click to browse"}
              </div>
            </div>

            {/* Selected File */}
            {state.file && (
              <div className="bg-primary/5 border-primary/10 mb-4 flex items-center gap-3 rounded-sm border px-3 py-3.5">
                <div className="text-primary">
                  <FileTextIcon />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="overflow-hidden text-[13px] font-medium text-ellipsis">
                    {state.file.name}
                  </div>
                  <div className="text-muted-foreground font-mono text-[11px]">
                    {state.isDemo ? "128 KB (demo)" : formatBytes(state.file.size)}
                  </div>
                </div>
                <button
                  className="text-muted-foreground cursor-pointer p-0.5"
                  onClick={() => setState((s) => ({ ...s, file: null, isDemo: false }))}
                >
                  <XIcon size={18} />
                </button>
              </div>
            )}

            {/* Buttons */}
            <div className="flex gap-2">
              <Button className="flex-1" disabled={!canParse} onClick={onParse}>
                <ArrowRightIcon /> Analyze CV
              </Button>
              <Button variant="outline" disabled={canParse} onClick={onDemo}>
                Demo
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* How it works */}
        <div className="flex flex-col gap-3">
          {PROCESS_STEPS.map((item, idx) => (
            <StepCard key={idx} item={item} index={idx} />
          ))}
        </div>
      </div>
    </div>
  );
}

import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { checkHealth } from "@/lib/api";
import type { Health } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ActivityIcon, AlertCircleIcon } from "lucide-react";
import { memo, useEffect, useState, type ReactNode } from "react";

type Status = "loading" | "ok" | "error";

const STATUS_CONFIG: Record<Status, { label: string; color: string; icon: ReactNode }> = {
  loading: {
    label: "Checking Server...",
    color: "border-chart-4/20 bg-chart-4/10 text-chart-4",
    icon: <ActivityIcon size={14} />,
  },
  ok: {
    label: "Server Online",
    color: "border-chart-3/20 bg-chart-3/10 text-chart-3",
    icon: (
      <span className="relative flex h-2 w-2">
        <span className="bg-chart-3 absolute h-full w-full animate-ping rounded-full opacity-75" />
        <span className="bg-chart-3 relative h-full w-full rounded-full" />
      </span>
    ),
  },
  error: {
    label: "Server Offline",
    color: "border-chart-5/20 bg-chart-5/10 text-chart-5",
    icon: <AlertCircleIcon size={14} />,
  },
};

const STATUS_DISPLAY: Record<Status, string> = {
  ok: "Online",
  loading: "Checking...",
  error: "Offline",
};

export const BackendStatus = memo(function BackendStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const check = async () => {
      try {
        setHealth(await checkHealth());
      } catch {
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };

    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  const status: Status = loading ? "loading" : health?.status === "ok" ? "ok" : "error";
  const { label, color, icon } = STATUS_CONFIG[status];

  return (
    <HoverCard>
      <HoverCardTrigger>
        <div
          className={cn(
            "flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium tracking-wide",
            color,
          )}
        >
          {icon}
          {label}
        </div>
      </HoverCardTrigger>
      <HoverCardContent className="w-56 px-4 py-3">
        <div className="text-muted-foreground mb-2.5 text-[11px] font-medium tracking-widest uppercase">
          Backend
        </div>
        <div className="space-y-2">
          <Row label="Status">
            <span
              className={cn(
                "font-medium capitalize",
                status === "ok" ? "text-primary" : "text-destructive",
              )}
            >
              {STATUS_DISPLAY[status]}
            </span>
          </Row>
          <Row label="Model Loaded">{health?.model_loaded ? "Yes" : "No"}</Row>
          <Row label="Jobs Indexed">{health?.jobs_indexed ?? 0}</Row>
          <Row label="Ready">{health?.ready ? "Yes" : "No"}</Row>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
});

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-muted-foreground text-sm">{label}</span>
      <span className="text-sm">{children}</span>
    </div>
  );
}

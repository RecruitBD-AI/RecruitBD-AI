import { cn } from "@/lib/utils";
import { VIEWS } from "@/lib/types";
import { CheckIcon } from "lucide-react";
import { memo } from "react";

export const STEP_INDEX = Object.fromEntries(VIEWS.map((v, i) => [v, i])) as Record<
  (typeof VIEWS)[number],
  number
>;

export const StepBar = memo(function StepBar({ current }: { current: number }) {
  return (
    <div className="flex items-center gap-0">
      {VIEWS.map((s, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold transition-all",
                i <= current ? "bg-primary text-background" : "bg-border",
              )}
            >
              {i < current ? <CheckIcon size={16} /> : i + 1}
            </div>
            <span
              className={cn(
                "text-xs transition-colors first-letter:uppercase",
                i === current ? "font-semibold" : "text-muted-foreground",
              )}
            >
              {s}
            </span>
          </div>
          {i < VIEWS.length - 1 && (
            <div
              className={cn(
                "mr-2 h-px w-8 transition-colors",
                i < current ? "bg-primary" : "bg-border",
              )}
            ></div>
          )}
        </div>
      ))}
    </div>
  );
});

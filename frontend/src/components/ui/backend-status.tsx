import { checkHealth } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ActivityIcon, AlertCircleIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";

type Status = "loading" | "ok" | "error";

const statusText: Record<Status, string> = {
  loading: "Checking Server...",
  error: "Server Offline",
  ok: "Server Online",
};

export const BackendStatus = memo(function BackendStatus() {
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const data = await checkHealth();
        setStatus(data.status);
      } catch (err) {
        setStatus("error");
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium tracking-wide",
        status === "ok" && "border-chart-3/20 bg-chart-3/10 text-chart-3",
        status === "loading" && "border-chart-4/20 bg-chart-4/10 text-chart-4",
        status === "error" && "border-chart-5/20 bg-chart-5/10 text-chart-5",
      )}
    >
      {status === "ok" && (
        <>
          <span className="relative flex h-2 w-2">
            <span className="bg-chart-3 absolute h-full w-full animate-ping rounded-full opacity-75"></span>
            <span className="bg-chart-3 relative h-full w-full rounded-full"></span>
          </span>
          {statusText[status]}
        </>
      )}
      {status === "loading" && (
        <>
          <ActivityIcon size={14} />
          {statusText[status]}
        </>
      )}
      {status === "error" && (
        <>
          <AlertCircleIcon size={14} />
          {statusText[status]}
        </>
      )}
    </div>
  );
});

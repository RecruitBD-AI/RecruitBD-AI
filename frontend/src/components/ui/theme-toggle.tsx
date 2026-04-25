import { useTheme } from "@/components/theme-provider";
import { cn } from "@/lib/utils";
import { MonitorIcon, MoonIcon, SunIcon } from "lucide-react";
import { memo } from "react";

export const ThemeToggle = memo(function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  const options = [
    { value: "light", label: "Light", icon: SunIcon },
    { value: "dark", label: "Dark", icon: MoonIcon },
    { value: "system", label: "System", icon: MonitorIcon },
  ] as const;

  return (
    <div className="bg-muted flex items-center gap-0.5 rounded-sm p-1">
      {options.map(({ value, label, icon: Icon }) => {
        const active = theme === value;

        return (
          <button
            key={value}
            onClick={() => setTheme(value)}
            title={label}
            className={cn(
              "item-center flex rounded p-1",
              active
                ? "bg-background dark:bg-border text-foreground shadow-sm"
                : "text-muted-foreground hover:bg-background/50",
            )}
          >
            <Icon size={16} />
            <span className="hidden">{label}</span>
          </button>
        );
      })}
    </div>
  );
});

import logo from "@/assets/logo.png";
import { cn } from "@/lib/utils";

/**
 * App brand mark — the basketball + trend-arrow icon (navy badge baked in).
 * Imported from src/assets so Vite hashes + cache-busts it.
 */
export function Logo({
  size = 36,
  className,
  glow = true,
}: {
  size?: number;
  className?: string;
  glow?: boolean;
}) {
  return (
    <img
      src={logo}
      alt="NBA Predict logo"
      width={size}
      height={size}
      loading="eager"
      decoding="async"
      style={{ width: size, height: size }}
      className={cn(
        "rounded-xl ring-1 ring-white/10",
        glow && "shadow-lg shadow-primary/25",
        className,
      )}
    />
  );
}

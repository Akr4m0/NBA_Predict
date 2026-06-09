import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
}

export function GlassCard({ children, className }: GlassCardProps) {
  return (
    <div
      className={cn(
        "backdrop-blur-md bg-white/5 border border-white/10 rounded-2xl p-6",
        className
      )}
    >
      {children}
    </div>
  );
}

import { Database, type LucideIcon } from "lucide-react";

interface EmptyStateProps {
  message: string;
  icon?: LucideIcon;
}

export function EmptyState({ message, icon: Icon = Database }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 p-12 text-center">
      <Icon className="mb-4 h-12 w-12 text-white/30" />
      <p className="text-sm text-white/50">{message}</p>
    </div>
  );
}

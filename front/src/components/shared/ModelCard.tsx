import { motion } from "framer-motion";
import { LucideIcon, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface ModelCardProps {
  name: string;
  description: string;
  icon: LucideIcon;
  accuracy?: number;
  isSelected?: boolean;
  onSelect?: () => void;
  delay?: number;
}

export function ModelCard({
  name,
  description,
  icon: Icon,
  accuracy,
  isSelected,
  onSelect,
  delay = 0,
}: ModelCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ scale: 1.02, y: -4 }}
      onClick={onSelect}
      className={cn(
        "relative cursor-pointer p-6 rounded-xl border transition-all duration-300",
        isSelected
          ? "bg-primary/10 border-primary shadow-lg shadow-primary/20"
          : "bg-card/50 border-border/50 hover:bg-card/70 hover:border-primary/30"
      )}
    >
      {isSelected && (
        <div className="absolute top-3 right-3 flex h-6 w-6 items-center justify-center rounded-full bg-primary">
          <Check className="h-4 w-4 text-primary-foreground" />
        </div>
      )}
      
      <div className="flex items-start gap-4">
        <div className={cn(
          "flex h-12 w-12 items-center justify-center rounded-lg transition-all duration-300",
          isSelected ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
        )}>
          <Icon className="h-6 w-6" />
        </div>
        
        <div className="flex-1 space-y-1">
          <h3 className="font-semibold">{name}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
          {accuracy !== undefined && (
            <div className="flex items-center gap-2 pt-2">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-success transition-all duration-500"
                  style={{ width: `${accuracy}%` }}
                />
              </div>
              <span className="text-sm font-medium stats-text">{accuracy}%</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

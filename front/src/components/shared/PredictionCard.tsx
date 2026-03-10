import { motion } from "framer-motion";
import { Calendar, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface PredictionCardProps {
  homeTeam: string;
  awayTeam: string;
  predictedWinner: "home" | "away";
  confidence: number;
  date: string;
  model: string;
  delay?: number;
}

export function PredictionCard({
  homeTeam,
  awayTeam,
  predictedWinner,
  confidence,
  date,
  model,
  delay = 0,
}: PredictionCardProps) {
  const getConfidenceColor = (conf: number) => {
    if (conf >= 70) return "text-success";
    if (conf >= 50) return "text-warning";
    return "text-destructive";
  };

  const getConfidenceBg = (conf: number) => {
    if (conf >= 70) return "bg-success";
    if (conf >= 50) return "bg-warning";
    return "bg-destructive";
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ scale: 1.02 }}
      className="glass-card-hover p-6 space-y-4"
    >
      {/* Date and Model */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Calendar className="h-4 w-4" />
          <span>{date}</span>
        </div>
        <span className="rounded-full bg-secondary/20 px-3 py-1 text-xs font-medium text-secondary">
          {model}
        </span>
      </div>

      {/* Teams */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex-1 text-center">
          <p className={cn(
            "text-lg font-bold",
            predictedWinner === "home" ? "text-primary" : "text-foreground"
          )}>
            {homeTeam}
          </p>
          <p className="text-xs text-muted-foreground">HOME</p>
        </div>

        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted">
          <span className="text-sm font-bold text-muted-foreground">VS</span>
        </div>

        <div className="flex-1 text-center">
          <p className={cn(
            "text-lg font-bold",
            predictedWinner === "away" ? "text-primary" : "text-foreground"
          )}>
            {awayTeam}
          </p>
          <p className="text-xs text-muted-foreground">AWAY</p>
        </div>
      </div>

      {/* Confidence */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className={cn("h-4 w-4", getConfidenceColor(confidence))} />
            <span className="text-sm text-muted-foreground">Confidence</span>
          </div>
          <span className={cn("font-bold stats-text", getConfidenceColor(confidence))}>
            {confidence}%
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-muted">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ duration: 1, delay: delay + 0.3 }}
            className={cn("h-full rounded-full", getConfidenceBg(confidence))}
          />
        </div>
      </div>

      {/* Predicted Winner */}
      <div className="rounded-lg bg-primary/10 p-3 text-center">
        <p className="text-xs text-muted-foreground">Predicted Winner</p>
        <p className="text-lg font-bold text-primary">
          {predictedWinner === "home" ? homeTeam : awayTeam}
        </p>
      </div>
    </motion.div>
  );
}

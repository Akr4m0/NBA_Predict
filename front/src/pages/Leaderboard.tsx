import { useState } from "react";
import { motion } from "framer-motion";
import { Trophy, Medal, Award, Database } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Skeleton } from "@/components/ui/skeleton";
import { GlassCard } from "@/components/ui/GlassCard";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getLeaderboard, getImports, getFreshnessStatus } from "@/lib/api";
import type { LeaderboardEntry, LeaderboardSort } from "@/lib/api";
import { cn } from "@/lib/utils";

// ── Sort options ──────────────────────────────────────────────────────────

const SORTS: { key: LeaderboardSort; label: string }[] = [
  { key: "accuracy", label: "Test Accuracy" },
  { key: "cv", label: "CV Accuracy" },
  { key: "f1", label: "F1 Score" },
];

// The metric column each sort ranks on (matches backend `sorted_by`).
const SORT_VALUE: Record<LeaderboardSort, (e: LeaderboardEntry) => number | null> = {
  accuracy: (e) => e.accuracy,
  cv: (e) => e.cv_accuracy_mean,
  f1: (e) => e.f1_score,
};

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

// ── Rank badge ────────────────────────────────────────────────────────────

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1)
    return <Medal className="h-6 w-6 text-yellow-400" aria-label="1st" />;
  if (rank === 2)
    return <Medal className="h-6 w-6 text-slate-300" aria-label="2nd" />;
  if (rank === 3)
    return <Award className="h-6 w-6 text-amber-600" aria-label="3rd" />;
  return (
    <span className="flex h-6 w-6 items-center justify-center text-sm font-semibold tabular-nums text-muted-foreground">
      {rank}
    </span>
  );
}

// ── Leaderboard row ───────────────────────────────────────────────────────

function LeaderRow({
  entry,
  sort,
  best,
  index,
}: {
  entry: LeaderboardEntry;
  sort: LeaderboardSort;
  best: number;
  index: number;
}) {
  const primary = SORT_VALUE[sort](entry);
  const widthPct = best > 0 && primary != null ? (primary / best) * 100 : 0;
  const isLeader = entry.rank === 1;

  return (
    <motion.div
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, delay: 0.1 + index * 0.05 }}
      className={cn(
        "rounded-xl border p-4",
        isLeader ? "border-orange-500/40 bg-orange-500/5" : "border-white/10 bg-white/5",
      )}
    >
      <div className="flex items-center gap-4">
        <RankBadge rank={entry.rank} />

        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate font-semibold">{entry.model_name ?? `Model ${entry.model_id}`}</p>
              <p className="text-xs capitalize text-muted-foreground">
                {(entry.model_type ?? "").replace(/_/g, " ")}
                {entry.evaluated_games > 0 && (
                  <span className="ml-2">· {entry.evaluated_games.toLocaleString()} games</span>
                )}
              </p>
            </div>
            <span
              className={cn(
                "shrink-0 text-2xl font-bold tabular-nums",
                isLeader ? "text-orange-400" : "text-white",
              )}
            >
              {pct(primary)}
            </span>
          </div>

          {/* Bar scaled to the leader */}
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${widthPct}%` }}
              transition={{ duration: 0.7, delay: 0.2 + index * 0.05, ease: "easeOut" }}
              className={cn(
                "h-full rounded-full",
                isLeader
                  ? "bg-gradient-to-r from-orange-500 to-orange-300"
                  : "bg-gradient-to-r from-orange-500/60 to-orange-300/60",
              )}
            />
          </div>

          {/* Secondary metrics */}
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs tabular-nums text-muted-foreground">
            <span>Acc <span className="font-medium text-white/80">{pct(entry.accuracy)}</span></span>
            <span>
              CV{" "}
              <span className="font-medium text-white/80">
                {pct(entry.cv_accuracy_mean)}
                {entry.cv_accuracy_std != null && (
                  <span className="text-white/40"> ±{(entry.cv_accuracy_std * 100).toFixed(1)}</span>
                )}
              </span>
            </span>
            <span>F1 <span className="font-medium text-white/80">{entry.f1_score != null ? entry.f1_score.toFixed(3) : "—"}</span></span>
            <span>Prec <span className="font-medium text-white/80">{pct(entry.precision)}</span></span>
            <span>Rec <span className="font-medium text-white/80">{pct(entry.recall)}</span></span>
            {entry.calibration_method && (
              <span className="capitalize">Calib <span className="font-medium text-white/80">{entry.calibration_method}</span></span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────

export default function Leaderboard() {
  const [sort, setSort] = useState<LeaderboardSort>("accuracy");
  const [importScope, setImportScope] = useState<string>("all");

  const { data: imports = [] } = useQuery({ queryKey: ["imports"], queryFn: getImports });
  const { data: freshness } = useQuery({
    queryKey: ["freshness-status"],
    queryFn: getFreshnessStatus,
  });

  const importId = importScope === "all" ? undefined : Number(importScope);
  const { data: entries = [], isLoading } = useQuery({
    queryKey: ["leaderboard", sort, importScope],
    queryFn: () => getLeaderboard(sort, undefined, importId),
  });

  const best = entries.reduce(
    (m, e) => Math.max(m, SORT_VALUE[sort](e) ?? 0),
    0,
  );

  return (
    <div className="min-h-screen">
      <main className="container mx-auto px-4 pb-12 pt-24">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 flex flex-wrap items-end justify-between gap-4"
        >
          <div>
            <h1 className="flex items-center gap-3 text-3xl font-bold md:text-4xl">
              <Trophy className="h-8 w-8 text-orange-400" />
              Leaderboard
            </h1>
            <p className="mt-2 text-muted-foreground">
              Every trained model ranked by performance. CV accuracy is the most
              honest metric — it's the temporal cross-validation score.
            </p>
            {freshness && (
              <p className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
                <Database className="h-3.5 w-3.5" />
                Data through{" "}
                <span className="font-medium text-foreground">
                  {freshness.latest_game_date ?? "—"}
                </span>
                · {freshness.total_games.toLocaleString()} games ·{" "}
                <span className={freshness.configured ? "text-green-400" : "text-muted-foreground"}>
                  {freshness.configured ? "live feed connected" : "live feed off"}
                </span>
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Import scope — keeps cross-import (different test split) accuracies separate */}
            <Select value={importScope} onValueChange={setImportScope}>
              <SelectTrigger className="w-44 border-white/10 bg-white/5">
                <SelectValue placeholder="All imports" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All imports</SelectItem>
                {imports.map((imp) => (
                  <SelectItem key={imp.id} value={String(imp.id)}>
                    {imp.filename} (#{imp.id})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Sort toggle */}
            <div className="inline-flex rounded-lg border border-white/10 bg-white/5 p-1">
              {SORTS.map((s) => (
                <button
                  key={s.key}
                  onClick={() => setSort(s.key)}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500",
                    sort === s.key
                      ? "bg-primary/15 text-primary"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Body */}
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-28 w-full rounded-xl" />
            ))}
          </div>
        ) : entries.length === 0 ? (
          <EmptyState message="No trained models yet. Train a model on the Train page to populate the leaderboard." />
        ) : (
          <GlassCard className="space-y-3">
            {entries.map((e, i) => (
              <LeaderRow key={e.model_id} entry={e} sort={sort} best={best} index={i} />
            ))}
          </GlassCard>
        )}
      </main>
    </div>
  );
}

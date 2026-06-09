import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, Target, Layers } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { MetricCard } from "@/components/shared/MetricCard";
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
import { getImports, getVerification } from "@/lib/api";
import type { VerifyPerModel } from "@/lib/api";
import { cn } from "@/lib/utils";

// ── Confusion Matrix ──────────────────────────────────────────────────────

function ConfusionMatrix({ model }: { model: VerifyPerModel }) {
  const tp = model.tp ?? 0;
  const tn = model.tn ?? 0;
  const fp = model.fp ?? 0;
  const fn = model.fn ?? 0;
  const total = tp + tn + fp + fn;

  const maxVal = Math.max(tp, tn, fp, fn) || 1;
  const intensity = (v: number) => 0.08 + 0.22 * (v / maxVal);

  const cells = [
    {
      label: "True Positive",
      value: tp,
      good: true,
      hint: "Predicted Home Win → was Home Win",
    },
    {
      label: "False Positive",
      value: fp,
      good: false,
      hint: "Predicted Home Win → was Away Win",
    },
    {
      label: "False Negative",
      value: fn,
      good: false,
      hint: "Predicted Away Win → was Home Win",
    },
    {
      label: "True Negative",
      value: tn,
      good: true,
      hint: "Predicted Away Win → was Away Win",
    },
  ];

  const precision = tp + fp > 0 ? tp / (tp + fp) : null;
  const recall = tp + fn > 0 ? tp / (tp + fn) : null;
  const accuracy = total > 0 ? (tp + tn) / total : null;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        {cells.map((c) => (
          <div
            key={c.label}
            className={cn(
              "rounded-xl p-4",
              c.good ? "border border-green-500/30" : "border border-red-500/30",
            )}
            style={{
              background: c.good
                ? `rgba(34, 197, 94, ${intensity(c.value)})`
                : `rgba(239, 68, 68, ${intensity(c.value)})`,
            }}
          >
            <p
              className={cn(
                "text-2xl font-bold tabular-nums",
                c.good ? "text-green-300" : "text-red-300",
              )}
            >
              {c.value.toLocaleString()}
            </p>
            <p
              className={cn(
                "text-xs font-medium",
                c.good ? "text-green-300" : "text-red-300",
              )}
            >
              {c.label}
            </p>
            <p className="mt-0.5 text-xs text-white/40">{c.hint}</p>
          </div>
        ))}
      </div>
      <div className="flex justify-around text-xs tabular-nums text-muted-foreground">
        <span>
          Precision:{" "}
          <span className="font-medium text-white/80">
            {precision != null ? `${(precision * 100).toFixed(1)}%` : "—"}
          </span>
        </span>
        <span>
          Recall:{" "}
          <span className="font-medium text-white/80">
            {recall != null ? `${(recall * 100).toFixed(1)}%` : "—"}
          </span>
        </span>
        <span>
          Accuracy:{" "}
          <span className="font-medium text-white/80">
            {accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : "—"}
          </span>
        </span>
      </div>
    </div>
  );
}

// ── Per-model accuracy bar row ───────────────────────────────────────────

function ModelAccuracyRow({
  model,
  index,
  best,
}: {
  model: VerifyPerModel;
  index: number;
  best: number;
}) {
  const pct = (model.accuracy * 100).toFixed(1);
  const widthPct = best > 0 ? (model.accuracy / best) * 100 : 0;
  const isBest = model.accuracy === best;

  return (
    <motion.div
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, delay: 0.4 + index * 0.05 }}
      className="space-y-1.5"
    >
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate font-medium">{model.model_name}</span>
        <span className="shrink-0 tabular-nums text-muted-foreground">
          <span className={cn("font-semibold", isBest ? "text-green-400" : "text-white")}>
            {pct}%
          </span>
          <span className="ml-2 text-xs">
            {model.correct.toLocaleString()} / {model.total.toLocaleString()}
          </span>
        </span>
      </div>
      <div className="relative h-2 overflow-hidden rounded-full bg-white/5">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${widthPct}%` }}
          transition={{ duration: 0.7, delay: 0.45 + index * 0.05, ease: "easeOut" }}
          className={cn(
            "h-full rounded-full",
            isBest
              ? "bg-gradient-to-r from-orange-500 to-orange-300"
              : "bg-gradient-to-r from-orange-500/70 to-orange-300/70",
          )}
        />
      </div>
    </motion.div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────

export default function Verify() {
  const [importId, setImportId] = useState<string>("");

  const { data: imports = [], isLoading: importsLoading } = useQuery({
    queryKey: ["imports"],
    queryFn: getImports,
  });

  const { data: verify, isLoading: verifyLoading, isFetching: verifyFetching } = useQuery({
    queryKey: ["verify", importId],
    queryFn: () => getVerification(Number(importId)),
    enabled: !!importId,
  });

  const perModel = verify?.per_model ?? [];
  const sortedModels = [...perModel].sort((a, b) => b.accuracy - a.accuracy);
  const bestAccuracy = sortedModels[0]?.accuracy ?? 0;

  const overallAccuracyPct =
    verify && verify.total > 0 ? (verify.accuracy * 100).toFixed(1) : "0.0";
  const incorrect = verify ? verify.total - verify.correct : 0;

  return (
    <div className="min-h-screen">

      <main className="container mx-auto px-4 pb-12 pt-24">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold md:text-4xl">Verification</h1>
          <p className="mt-2 text-muted-foreground">
            Real test-set accuracy and confusion matrices for every trained model.
          </p>
        </motion.div>

        {/* Import selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8"
        >
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Import session</label>
            {importsLoading ? (
              <Skeleton className="h-10 w-64" />
            ) : imports.length === 0 ? (
              <EmptyState message="No imports yet. Upload a CSV on the Import page to get started." />
            ) : (
              <Select value={importId} onValueChange={setImportId}>
                <SelectTrigger className="w-64 border-border/50 bg-muted/30">
                  <SelectValue placeholder="Select an import…" />
                </SelectTrigger>
                <SelectContent>
                  {imports.map((imp) => (
                    <SelectItem key={imp.id} value={String(imp.id)}>
                      {imp.filename}
                      {imp.record_count != null && (
                        <span className="ml-2 text-xs text-muted-foreground">
                          ({imp.record_count.toLocaleString()} rows)
                        </span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </motion.div>

        {/* Body */}
        {!importId ? (
          imports.length === 0 ? null : (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Pick an import session above to load verification results.
            </p>
          )
        ) : verifyLoading || verifyFetching ? (
          <div className="space-y-8">
            <div className="grid gap-4 md:grid-cols-4">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-32 rounded-2xl" />
              ))}
            </div>
            <Skeleton className="h-64 w-full rounded-2xl" />
          </div>
        ) : !verify || verify.per_model.length === 0 ? (
          <EmptyState message="No verified predictions for this import session yet. Train a model on the Train page." />
        ) : (
          <>
            {/* KPI Row */}
            <div className="mb-8 grid gap-4 md:grid-cols-4">
              <MetricCard
                title="Total Games"
                value={verify.total.toLocaleString()}
                subtitle="Across all verified models"
                icon={Target}
                delay={0.15}
              />
              <MetricCard
                title="Correct"
                value={verify.correct.toLocaleString()}
                subtitle="Predictions matched outcome"
                icon={CheckCircle}
                delay={0.2}
              />
              <MetricCard
                title="Incorrect"
                value={incorrect.toLocaleString()}
                subtitle="Predictions missed"
                icon={XCircle}
                delay={0.25}
              />
              <MetricCard
                title="Overall Accuracy"
                value={`${overallAccuracyPct}%`}
                subtitle={`${verify.per_model.length} model${verify.per_model.length === 1 ? "" : "s"} verified`}
                icon={Layers}
                delay={0.3}
              />
            </div>

            <div className="grid gap-8 lg:grid-cols-2">
              {/* Per-model summary */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.35 }}
              >
                <GlassCard>
                  <h3 className="mb-4 font-semibold">Per-Model Accuracy</h3>
                  <div className="space-y-4">
                    {sortedModels.map((m, i) => (
                      <ModelAccuracyRow
                        key={m.model_name}
                        model={m}
                        index={i}
                        best={bestAccuracy}
                      />
                    ))}
                  </div>
                </GlassCard>
              </motion.div>

              {/* Confusion matrices */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <GlassCard className="space-y-6">
                  <div className="flex items-baseline justify-between">
                    <h3 className="font-semibold">Confusion Matrices</h3>
                    <p className="text-xs italic text-muted-foreground">
                      Ties treated as Away Win
                    </p>
                  </div>
                  {sortedModels.map((m) => (
                    <div key={m.model_name} className="space-y-2">
                      <p className="text-sm font-medium">{m.model_name}</p>
                      <ConfusionMatrix model={m} />
                    </div>
                  ))}
                </GlassCard>
              </motion.div>
            </div>
          </>
        )}
      </main>

    </div>
  );
}

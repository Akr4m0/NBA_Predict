import { useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
} from "recharts";
import { Award, Database } from "lucide-react";
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
import { getResults, getImports, getVerification } from "@/lib/api";
import type { ResultRecord } from "@/lib/api";

// ── Constants ─────────────────────────────────────────────────────────────

const RADAR_COLORS = ["#ff6b00", "#64748b", "#3b82f6", "#22c55e"];

// ── Confusion Matrix ──────────────────────────────────────────────────────

function ConfusionMatrix({
  correct,
  total,
  modelName,
}: {
  correct: number;
  total: number;
  modelName: string;
}) {
  const incorrect = total - correct;
  const homeWinRate = 0.58;
  const tp = Math.round(correct * homeWinRate);
  const tn = correct - tp;
  const fn = Math.round(incorrect * homeWinRate);
  const fp = incorrect - fn;

  const cells = [
    {
      label: "True Positive",
      value: tp,
      bg: "bg-green-500/30",
      text: "text-green-300",
      sub: "Home Win → correctly predicted",
    },
    {
      label: "False Positive",
      value: fp,
      bg: "bg-red-500/20",
      text: "text-red-300",
      sub: "Away Win → predicted as Home",
    },
    {
      label: "False Negative",
      value: fn,
      bg: "bg-red-500/20",
      text: "text-red-300",
      sub: "Home Win → predicted as Away",
    },
    {
      label: "True Negative",
      value: tn,
      bg: "bg-green-500/20",
      text: "text-green-300",
      sub: "Away Win → correctly predicted",
    },
  ];

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium">{modelName}</p>
      <div className="grid grid-cols-2 gap-2">
        {cells.map((c) => (
          <div key={c.label} className={`rounded-xl p-4 ${c.bg}`}>
            <p className={`text-2xl font-bold ${c.text}`}>
              {c.value.toLocaleString()}
            </p>
            <p className={`text-xs font-medium ${c.text}`}>{c.label}</p>
            <p className="mt-0.5 text-xs text-white/40">{c.sub}</p>
          </div>
        ))}
      </div>
      <p className="text-xs italic text-muted-foreground">
        * Estimated from {correct.toLocaleString()}/{total.toLocaleString()}{" "}
        correct. Assumes ~58% home-win base rate.
      </p>
    </div>
  );
}

// ── Feature importance chart ──────────────────────────────────────────────

function FeatureImportanceChart({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([feature, importance]) => ({ feature, importance }));

  if (!entries.length) return null;

  return (
    <ResponsiveContainer width="100%" height={entries.length * 32 + 20}>
      <BarChart
        data={entries}
        layout="vertical"
        margin={{ top: 0, right: 8, bottom: 0, left: 160 }}
      >
        <defs>
          <linearGradient id="featureGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#ff6b00" stopOpacity={0.9} />
            <stop offset="100%" stopColor="#ffbe99" stopOpacity={0.7} />
          </linearGradient>
        </defs>
        <XAxis
          type="number"
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={155}
        />
        <Tooltip
          formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
          contentStyle={{
            background: "rgba(10,10,20,0.9)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 8,
          }}
        />
        <Bar dataKey="importance" fill="url(#featureGrad)" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function Analysis() {
  const [selectedModel, setSelectedModel] = useState<string>("all");
  const [verifyImportId, setVerifyImportId] = useState<string>("");

  const { data: results = [], isLoading: resultsLoading } = useQuery({
    queryKey: ["results"],
    queryFn: () => getResults(),
  });

  const { data: imports = [], isLoading: importsLoading } = useQuery({
    queryKey: ["imports"],
    queryFn: getImports,
  });

  const { data: verifyData, isLoading: verifyLoading } = useQuery({
    queryKey: ["verify", verifyImportId],
    queryFn: () => getVerification(Number(verifyImportId)),
    enabled: !!verifyImportId,
  });

  const filtered: ResultRecord[] =
    selectedModel === "all"
      ? results
      : results.filter((r) => r.name === selectedModel);

  const bestModel = results.length
    ? results.reduce((best, r) => (r.accuracy > best.accuracy ? r : best))
    : null;

  // Radar data
  const radarData = ["Accuracy", "Precision", "Recall", "F1-Score"].map(
    (metric) => {
      const row: Record<string, unknown> = { metric };
      filtered.forEach((r) => {
        if (metric === "Accuracy") row[r.name] = r.accuracy ?? 0;
        else if (metric === "Precision") row[r.name] = r.precision_score ?? 0;
        else if (metric === "Recall") row[r.name] = r.recall ?? 0;
        else row[r.name] = r.f1_score ?? 0;
      });
      return row;
    }
  );

  // Feature importance
  const fiSource =
    selectedModel !== "all"
      ? results.find((r) => r.name === selectedModel)
      : bestModel;

  const featureImportance: Record<string, number> | null = (() => {
    if (!fiSource?.feature_importance) return null;
    try {
      const parsed = JSON.parse(fiSource.feature_importance);
      if (Array.isArray(parsed)) {
        const obj: Record<string, number> = {};
        for (const item of parsed) obj[item.feature] = item.importance;
        return obj;
      }
      return parsed as Record<string, number>;
    } catch {
      return null;
    }
  })();

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
          <h1 className="text-3xl font-bold md:text-4xl">Analysis</h1>
          <p className="mt-2 text-muted-foreground">
            Compare model performance and analyze prediction metrics
          </p>
        </motion.div>

        {/* Controls */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8"
        >
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Model</label>
            {resultsLoading ? (
              <Skeleton className="h-10 w-48" />
            ) : (
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="w-48 border-border/50 bg-muted/30">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Models</SelectItem>
                  {results.map((r) => (
                    <SelectItem key={r.name} value={r.name}>
                      {r.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </motion.div>

        {/* Best model banner */}
        {bestModel && !resultsLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="mb-8"
          >
            <GlassCard className="flex items-center gap-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-primary/10">
                <Award className="h-7 w-7 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">
                  Best Performing Model
                </p>
                <p className="text-2xl font-bold">
                  {bestModel.name}{" "}
                  <span className="text-xl text-green-400">
                    {(bestModel.accuracy * 100).toFixed(1)}% accuracy
                  </span>
                </p>
              </div>
            </GlassCard>
          </motion.div>
        )}

        {/* Main content */}
        {resultsLoading ? (
          <div className="grid gap-8 lg:grid-cols-2">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-80 w-full rounded-2xl" />
            ))}
          </div>
        ) : results.length === 0 ? (
          <EmptyState message="No model results yet. Train a model on the Train page to see analysis here." />
        ) : (
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Radar Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <GlassCard>
                <h3 className="mb-4 font-semibold">Model Comparison — Radar</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis
                      dataKey="metric"
                      tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
                    />
                    <PolarRadiusAxis
                      angle={30}
                      domain={[0, 1]}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
                    />
                    {filtered.map((r, i) => (
                      <Radar
                        key={r.name}
                        name={r.name}
                        dataKey={r.name}
                        stroke={RADAR_COLORS[i % RADAR_COLORS.length]}
                        fill={RADAR_COLORS[i % RADAR_COLORS.length]}
                        fillOpacity={0.15}
                        strokeWidth={2}
                      />
                    ))}
                    <Legend
                      wrapperStyle={{
                        fontSize: 12,
                        color: "rgba(255,255,255,0.6)",
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </GlassCard>
            </motion.div>

            {/* Feature Importance */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <GlassCard>
                <h3 className="mb-4 font-semibold">
                  Feature Importance
                  {fiSource && (
                    <span className="ml-2 text-sm font-normal text-muted-foreground">
                      ({fiSource.name})
                    </span>
                  )}
                </h3>
                {featureImportance ? (
                  <FeatureImportanceChart data={featureImportance} />
                ) : (
                  <div className="flex h-48 flex-col items-center justify-center text-center text-sm text-muted-foreground">
                    <p>No feature importance data available.</p>
                    <p className="text-xs mt-1">Train a tree-based model (RF, DT, XGBoost).</p>
                  </div>
                )}
              </GlassCard>
            </motion.div>

            {/* Confusion Matrix */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <GlassCard className="space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h3 className="font-semibold">Confusion Matrix</h3>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">
                      Import session
                    </label>
                    {importsLoading ? (
                      <Skeleton className="h-9 w-44" />
                    ) : (
                      <Select
                        value={verifyImportId}
                        onValueChange={setVerifyImportId}
                      >
                        <SelectTrigger className="h-9 w-44 border-border/50 bg-muted/30 text-xs">
                          <SelectValue placeholder="Select import…" />
                        </SelectTrigger>
                        <SelectContent>
                          {imports.map((imp) => (
                            <SelectItem key={imp.id} value={String(imp.id)}>
                              {imp.filename}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                </div>

                {!verifyImportId ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    Select an import session above to load confusion data.
                  </p>
                ) : verifyLoading ? (
                  <div className="space-y-4">
                    {[1, 2].map((i) => (
                      <div key={i} className="space-y-2">
                        <Skeleton className="h-4 w-32" />
                        <div className="grid grid-cols-2 gap-2">
                          {[1, 2, 3, 4].map((j) => (
                            <Skeleton key={j} className="h-20 rounded-xl" />
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : !verifyData || verifyData.per_model.length === 0 ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    No verification data for this import session.
                  </p>
                ) : (
                  <div className="space-y-6">
                    {verifyData.per_model.map((m) => (
                      <ConfusionMatrix
                        key={m.model_name}
                        correct={m.correct}
                        total={m.total}
                        modelName={m.model_name}
                      />
                    ))}
                    <p className="text-right text-xs italic text-muted-foreground">
                      Ties treated as Away Win.
                    </p>
                  </div>
                )}
              </GlassCard>
            </motion.div>

            {/* Metrics Summary */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <GlassCard>
                <h3 className="mb-4 font-semibold">Metrics Summary</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="pb-3 text-left font-medium text-muted-foreground">
                          Model
                        </th>
                        <th className="pb-3 text-right font-medium text-muted-foreground">
                          Accuracy
                        </th>
                        <th className="pb-3 text-right font-medium text-muted-foreground">
                          Precision
                        </th>
                        <th className="pb-3 text-right font-medium text-muted-foreground">
                          Recall
                        </th>
                        <th className="pb-3 text-right font-medium text-muted-foreground">
                          F1
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {filtered.map((r) => (
                        <tr
                          key={r.name}
                          className="transition-colors hover:bg-white/5"
                        >
                          <td className="py-3 font-medium">{r.name}</td>
                          <td className="py-3 text-right tabular-nums">
                            {r.accuracy != null
                              ? `${(r.accuracy * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="py-3 text-right tabular-nums text-muted-foreground">
                            {r.precision_score != null
                              ? `${(r.precision_score * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="py-3 text-right tabular-nums text-muted-foreground">
                            {r.recall != null
                              ? `${(r.recall * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="py-3 text-right tabular-nums text-muted-foreground">
                            {r.f1_score != null
                              ? `${(r.f1_score * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>
            </motion.div>
          </div>
        )}
      </main>

    </div>
  );
}

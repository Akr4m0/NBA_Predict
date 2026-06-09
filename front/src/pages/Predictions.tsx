import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, ChevronsUpDown, Database, Trophy, Bookmark } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getImports,
  getImportTeams,
  getModels,
  predictGame,
  savePrediction,
  getPredictions,
  getUserLeaderboard,
} from "@/lib/api";
import type { PredictResponse, ModelRecord } from "@/lib/api";
import { GlassCard } from "@/components/ui/GlassCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { getTeamName } from "@/lib/teams";

// ── Helpers ───────────────────────────────────────────────────────────────

function confidenceClass(conf: number) {
  if (conf >= 0.75) return "bg-green-500/20 text-green-300 border-green-500/30";
  if (conf >= 0.55) return "bg-amber-500/20 text-amber-300 border-amber-500/30";
  return "bg-red-500/20 text-red-300 border-red-500/30";
}

function confidenceLabel(conf: number) {
  if (conf >= 0.75) return "High confidence";
  if (conf >= 0.55) return "Medium confidence";
  return "Low confidence";
}

// ── Team Combobox ─────────────────────────────────────────────────────────

function TeamCombobox({
  teams,
  value,
  onChange,
  placeholder,
  disabled,
}: {
  teams: number[];
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          disabled={disabled}
          className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-muted/30 px-3 py-2 text-sm hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className={value ? "text-foreground" : "text-muted-foreground"}>
            {value ? getTeamName(value) : placeholder}
          </span>
          <ChevronsUpDown className="h-4 w-4 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[260px]" align="start">
        <Command>
          <CommandInput placeholder="Search team…" />
          <CommandList>
            <CommandEmpty>No teams found.</CommandEmpty>
            <CommandGroup>
              {teams.map((teamId) => {
                const idStr = String(teamId);
                const name = getTeamName(teamId);
                return (
                  <CommandItem
                    key={teamId}
                    // Combine name + id so search matches either; we re-derive
                    // the canonical id in onSelect rather than trusting `v`.
                    value={`${name} ${idStr}`}
                    onSelect={() => {
                      onChange(idStr === value ? "" : idStr);
                      setOpen(false);
                    }}
                  >
                    <Check
                      className={`mr-2 h-4 w-4 ${
                        value === idStr ? "opacity-100" : "opacity-0"
                      }`}
                    />
                    <span className="flex-1">{name}</span>
                    <span className="ml-2 text-xs text-muted-foreground tabular-nums">
                      {idStr}
                    </span>
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

// ── Model Radio Card ──────────────────────────────────────────────────────

function ModelRadioCard({
  model,
  selected,
  onSelect,
}: {
  model: ModelRecord;
  selected: boolean;
  onSelect: () => void;
}) {
  const acc = model.accuracy;
  return (
    <button
      onClick={onSelect}
      className={`w-full rounded-xl border p-4 text-left transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2 ${
        selected
          ? "border-primary bg-primary/10"
          : "border-white/10 bg-white/5 hover:border-white/20"
      }`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="font-medium text-sm">{model.name}</p>
          <p className="text-xs text-muted-foreground capitalize">
            {model.type.replace(/_/g, " ")}
          </p>
        </div>
        {acc != null && (
          <span className="text-xs font-medium text-muted-foreground bg-white/10 rounded-full px-2.5 py-0.5">
            {(acc * 100).toFixed(1)}% acc
          </span>
        )}
      </div>
    </button>
  );
}

// ── Result Card ───────────────────────────────────────────────────────────

function ResultCard({ result }: { result: PredictResponse }) {
  const isHomeWin = result.prediction === "Home Win";
  const conf = result.confidence;

  const featureData = result.feature_importance
    ? Object.entries(result.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([feature, importance]) => ({ feature, importance }))
    : null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-5"
    >
      {/* Winner */}
      <div className="flex items-center gap-3">
        <div
          className={`flex h-12 w-12 items-center justify-center rounded-xl ${
            isHomeWin ? "bg-green-500/20" : "bg-red-500/20"
          }`}
        >
          <Trophy className={`h-6 w-6 ${isHomeWin ? "text-green-400" : "text-red-400"}`} />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Prediction</p>
          <p
            className={`text-2xl font-bold ${
              isHomeWin ? "text-green-400" : "text-red-400"
            }`}
          >
            {result.prediction}
          </p>
        </div>
      </div>

      {/* Confidence badge */}
      <div className="flex items-center gap-3">
        <span
          className={`rounded-full border px-3 py-1 text-sm font-medium ${confidenceClass(conf)}`}
        >
          {(conf * 100).toFixed(1)}% — {confidenceLabel(conf)}
        </span>
      </div>

      {/* Baseline disclaimer */}
      {!result.confidence_reliable && (
        <p className="text-xs text-amber-400 italic">
          Baseline model — confidence score is not meaningful.
        </p>
      )}

      {/* Model used */}
      <p className="text-xs text-muted-foreground">
        Model: <span className="text-foreground">{result.model_name}</span>
      </p>

      {/* Feature importance */}
      {featureData && featureData.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Feature Importance (top 10)</p>
          <ResponsiveContainer width="100%" height={featureData.length * 28 + 20}>
            <BarChart
              data={featureData}
              layout="vertical"
              margin={{ top: 0, right: 8, bottom: 0, left: 140 }}
            >
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
                width={135}
              />
              <Tooltip
                formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                contentStyle={{
                  background: "rgba(10,10,20,0.9)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {featureData.map((_, i) => (
                  <Cell
                    key={i}
                    fill={`rgba(255,107,0,${1 - i * 0.08})`}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function Predictions() {
  const [importId, setImportId] = useState<string>("");
  const [homeTeam, setHomeTeam] = useState<string>("");
  const [awayTeam, setAwayTeam] = useState<string>("");
  const [season, setSeason] = useState<string>("2022");
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [latestResult, setLatestResult] = useState<PredictResponse | null>(null);
  // Handle identifies the user for saved predictions (honor-system MVP).
  // Persisted to localStorage so it survives refresh.
  const [handle, setHandle] = useState<string>(
    () => localStorage.getItem("nba_handle") ?? ""
  );

  const queryClient = useQueryClient();

  useEffect(() => {
    localStorage.setItem("nba_handle", handle);
  }, [handle]);

  const { data: imports = [], isLoading: importsLoading } = useQuery({
    queryKey: ["imports"],
    queryFn: getImports,
  });

  const { data: teamsData, isLoading: teamsLoading } = useQuery({
    queryKey: ["teams", importId],
    queryFn: () => getImportTeams(Number(importId)),
    enabled: !!importId,
  });

  const { data: models = [], isLoading: modelsLoading } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
  });

  const trimmedHandle = handle.trim();
  const { data: history = [] } = useQuery({
    queryKey: ["predictions", trimmedHandle],
    queryFn: () => getPredictions(trimmedHandle, 10),
    enabled: !!trimmedHandle,
  });

  // Per-handle resolved-prediction accuracy (filled by the freshness pipeline).
  const { data: userAcc = [] } = useQuery({
    queryKey: ["user-leaderboard", trimmedHandle],
    queryFn: () => getUserLeaderboard(trimmedHandle),
    enabled: !!trimmedHandle,
  });
  const myAccuracy = userAcc[0];

  const mutation = useMutation({
    mutationFn: predictGame,
    onSuccess: (data) => {
      setLatestResult(data);
      toast.success(`Prediction: ${data.prediction}`);
    },
    onError: () => {
      toast.error("Prediction failed. Check your inputs.");
    },
  });

  const saveMutation = useMutation({
    mutationFn: savePrediction,
    onSuccess: (data) => {
      toast.success(`Saved as ${trimmedHandle}: ${data.prediction}`);
      queryClient.invalidateQueries({ queryKey: ["predictions", trimmedHandle] });
    },
    onError: () => {
      toast.error("Could not save prediction.");
    },
  });

  const canPredict =
    homeTeam && awayTeam && season && selectedModelId != null && !mutation.isPending;

  const handlePredict = () => {
    if (!canPredict) return;
    mutation.mutate({
      home_team: homeTeam,
      away_team: awayTeam,
      season,
      model_id: selectedModelId!,
    });
  };

  const canSave =
    !!trimmedHandle && homeTeam && awayTeam && season && selectedModelId != null &&
    !saveMutation.isPending;

  const handleSave = () => {
    if (!canSave) return;
    saveMutation.mutate({
      user_handle: trimmedHandle,
      home_team: homeTeam,
      away_team: awayTeam,
      season,
      model_id: selectedModelId!,
    });
  };

  const teams = teamsData?.teams ?? [];

  return (
    <div className="min-h-screen">

      <main className="container mx-auto px-4 pt-24 pb-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold md:text-4xl">Predictions</h1>
          <p className="mt-2 text-muted-foreground">
            Predict NBA game outcomes using your trained models
          </p>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-5">
          {/* Left — form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="lg:col-span-3 space-y-6"
          >
            {/* Handle */}
            <GlassCard className="space-y-3">
              <div className="flex items-center gap-2">
                <Bookmark className="h-5 w-5 text-primary" />
                <h2 className="text-lg font-semibold">Your Handle</h2>
              </div>
              <p className="text-sm text-muted-foreground">
                Pick a nickname to save predictions under. No password — saved
                picks are tied to this handle.
              </p>
              <Input
                value={handle}
                onChange={(e) => setHandle(e.target.value)}
                placeholder="e.g. akram"
                maxLength={40}
                className="bg-muted/30 border-border/50 max-w-[260px]"
              />
            </GlassCard>

            {/* Import session */}
            <GlassCard className="space-y-4">
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <h2 className="text-lg font-semibold">Import Session</h2>
              </div>
              {importsLoading ? (
                <Skeleton className="h-10 w-full" />
              ) : imports.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No imports found.{" "}
                  <a href="/import" className="text-primary underline">
                    Import data first.
                  </a>
                </p>
              ) : (
                <Select value={importId} onValueChange={(v) => { setImportId(v); setHomeTeam(""); setAwayTeam(""); }}>
                  <SelectTrigger className="bg-muted/30 border-border/50">
                    <SelectValue placeholder="Choose import session…" />
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
            </GlassCard>

            {/* Teams + season */}
            <GlassCard className="space-y-4">
              <h2 className="text-lg font-semibold">Matchup</h2>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Home Team</label>
                  {teamsLoading ? (
                    <Skeleton className="h-10 w-full" />
                  ) : (
                    <TeamCombobox
                      teams={teams}
                      value={homeTeam}
                      onChange={setHomeTeam}
                      placeholder="Select home team…"
                      disabled={!importId || teams.length === 0}
                    />
                  )}
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Away Team</label>
                  {teamsLoading ? (
                    <Skeleton className="h-10 w-full" />
                  ) : (
                    <TeamCombobox
                      teams={teams}
                      value={awayTeam}
                      onChange={setAwayTeam}
                      placeholder="Select away team…"
                      disabled={!importId || teams.length === 0}
                    />
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Season</label>
                <Input
                  value={season}
                  onChange={(e) => setSeason(e.target.value)}
                  placeholder="e.g. 2022"
                  className="bg-muted/30 border-border/50 max-w-[160px]"
                />
              </div>
            </GlassCard>

            {/* Model selector */}
            <GlassCard className="space-y-4">
              <h2 className="text-lg font-semibold">Select Model</h2>
              {modelsLoading ? (
                <div className="space-y-2">
                  {[1, 2].map((i) => <Skeleton key={i} className="h-16 w-full rounded-xl" />)}
                </div>
              ) : models.length === 0 ? (
                <EmptyState message="No trained models yet. Go to Train to train a model first." />
              ) : (
                <div className="grid gap-2 sm:grid-cols-2">
                  {models.map((m) => (
                    <ModelRadioCard
                      key={m.id}
                      model={m}
                      selected={selectedModelId === m.id}
                      onSelect={() => setSelectedModelId(m.id)}
                    />
                  ))}
                </div>
              )}
            </GlassCard>

            {/* Predict + Save buttons */}
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                onClick={handlePredict}
                disabled={!canPredict}
                variant="hero"
                size="lg"
                className="flex-1"
              >
                {mutation.isPending ? "Predicting…" : "Predict"}
              </Button>
              <Button
                onClick={handleSave}
                disabled={!canSave}
                variant="outline"
                size="lg"
                className="flex-1"
                title={!trimmedHandle ? "Enter a handle to save" : undefined}
              >
                <Bookmark className="mr-2 h-4 w-4" />
                {saveMutation.isPending ? "Saving…" : "Save Prediction"}
              </Button>
            </div>
          </motion.div>

          {/* Right — result + history */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Result card */}
            <AnimatePresence mode="wait">
              {latestResult ? (
                <ResultCard key="result" result={latestResult} />
              ) : (
                <motion.div
                  key="placeholder"
                  className="rounded-2xl border border-dashed border-white/10 p-8 text-center text-muted-foreground"
                >
                  <Trophy className="mx-auto mb-3 h-10 w-10 opacity-20" />
                  <p className="text-sm">Your prediction will appear here.</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* History table — saved predictions for the current handle */}
            {trimmedHandle && history.length > 0 && (
              <GlassCard className="space-y-3 p-5">
                <div className="flex items-baseline justify-between gap-2">
                  <h3 className="font-semibold text-sm">
                    {trimmedHandle}'s Saved Predictions
                  </h3>
                  {myAccuracy && myAccuracy.resolved_count > 0 && (
                    <span className="text-xs tabular-nums text-muted-foreground">
                      <span className="font-semibold text-foreground">
                        {((myAccuracy.accuracy ?? 0) * 100).toFixed(0)}%
                      </span>{" "}
                      ({myAccuracy.correct_count}/{myAccuracy.resolved_count} resolved)
                    </span>
                  )}
                </div>
                <div className="divide-y divide-white/5">
                  {history.map((h) => {
                    const isHome = h.prediction === "Home Win";
                    const conf = h.predicted_confidence ?? 0;
                    const resolved = h.resolved === 1;
                    const correct = h.correct === 1;
                    return (
                      <div key={h.id} className="flex items-center justify-between gap-2 py-2">
                        <div className="flex min-w-0 items-center gap-2">
                          <span
                            className={`h-2 w-2 shrink-0 rounded-full ${
                              isHome ? "bg-green-400" : "bg-red-400"
                            }`}
                          />
                          <div className="min-w-0">
                            <span className="block truncate text-sm">
                              {getTeamName(h.home_team)} vs {getTeamName(h.away_team)}
                            </span>
                            <span className="block text-xs text-muted-foreground">
                              {h.prediction}
                            </span>
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          {/* Resolution status — filled by the freshness pipeline */}
                          {resolved ? (
                            <span
                              title={`Actual: ${h.actual_label === "home_win" ? "Home Win" : "Away Win"}`}
                              className={`text-xs font-semibold ${
                                correct ? "text-green-400" : "text-red-400"
                              }`}
                            >
                              {correct ? "✓ Hit" : "✗ Miss"}
                            </span>
                          ) : (
                            <span className="text-xs text-muted-foreground/60">Pending</span>
                          )}
                          <span
                            className={`text-xs rounded-full px-2 py-0.5 ${confidenceClass(conf)}`}
                          >
                            {(conf * 100).toFixed(0)}%
                          </span>
                          <span className="hidden text-xs text-muted-foreground truncate max-w-[80px] sm:inline">
                            {h.model_name}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </GlassCard>
            )}
          </motion.div>
        </div>
      </main>

    </div>
  );
}

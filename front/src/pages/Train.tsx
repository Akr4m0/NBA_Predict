import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  ChevronDown,
  Database,
  Play,
  TreeDeciduous,
  Trees,
  Zap,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getImports, trainModel } from "@/lib/api";
import type { TrainResponse } from "@/lib/api";
import { GlassCard } from "@/components/ui/GlassCard";
import { EmptyState } from "@/components/ui/EmptyState";

// ── Types ─────────────────────────────────────────────────────────────────

type ModelType = "decision_tree" | "random_forest" | "xgboost";

interface ModelResult {
  accuracy: number;
  f1_score: number;
}

interface DTParams {
  max_depth: number;
  min_samples_split: number;
}
interface RFParams {
  n_estimators: number;
  max_depth: number;
}
interface XGBParams {
  n_estimators: number;
  learning_rate: number;
  max_depth: number;
}

// ── Model definitions ─────────────────────────────────────────────────────

const MODEL_DEFS = [
  {
    type: "decision_tree" as ModelType,
    name: "Decision Tree",
    icon: TreeDeciduous,
    description: "Interpretable decision paths with clear feature importance.",
    accuracyRange: "57–61%",
  },
  {
    type: "random_forest" as ModelType,
    name: "Random Forest",
    icon: Trees,
    description: "Ensemble of decision trees for improved robustness.",
    accuracyRange: "60–63%",
  },
  {
    type: "xgboost" as ModelType,
    name: "XGBoost",
    icon: Zap,
    description: "Gradient boosting — typically the strongest performer.",
    accuracyRange: "61–64%",
  },
] as const;

// ── Param defaults ────────────────────────────────────────────────────────

const DEFAULT_DT: DTParams = { max_depth: 10, min_samples_split: 2 };
const DEFAULT_RF: RFParams = { n_estimators: 100, max_depth: 10 };
const DEFAULT_XGB: XGBParams = { n_estimators: 100, learning_rate: 0.1, max_depth: 6 };

// ── ModelTrainCard ────────────────────────────────────────────────────────

function SliderRow({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v: number) => String(v),
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-medium text-primary">{format(value)}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      />
    </div>
  );
}

interface ModelTrainCardProps {
  def: (typeof MODEL_DEFS)[number];
  isTraining: boolean;
  elapsed: number;
  result: ModelResult | null;
  disabled: boolean;
  onTrain: () => void;
  // param state
  dtParams: DTParams;
  rfParams: RFParams;
  xgbParams: XGBParams;
  setDtParams: (p: DTParams) => void;
  setRfParams: (p: RFParams) => void;
  setXgbParams: (p: XGBParams) => void;
}

function ModelTrainCard({
  def,
  isTraining,
  elapsed,
  result,
  disabled,
  onTrain,
  dtParams,
  rfParams,
  xgbParams,
  setDtParams,
  setRfParams,
  setXgbParams,
}: ModelTrainCardProps) {
  const [open, setOpen] = useState(false);
  const Icon = def.icon;

  const accuracyColor =
    result && result.accuracy * 100 >= 65 ? "text-green-400" : "text-amber-400";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative rounded-2xl border p-5 space-y-4 transition-all duration-300 ${
        isTraining
          ? "border-primary/60 bg-primary/5"
          : "border-white/10 bg-white/5 hover:border-white/20"
      }`}
    >
      {/* Pulsing ring while training */}
      {isTraining && (
        <span className="absolute inset-0 rounded-2xl animate-ping border-2 border-primary opacity-30 pointer-events-none" />
      )}

      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="font-semibold">{def.name}</p>
            <span className="text-xs text-muted-foreground bg-white/10 px-2 py-0.5 rounded-full">
              {def.accuracyRange} typical
            </span>
          </div>
        </div>

        {/* Result chip */}
        {result && !isTraining && (
          <span className={`text-xs font-medium ${accuracyColor}`}>
            {(result.accuracy * 100).toFixed(1)}% acc · {(result.f1_score * 100).toFixed(1)}% F1
          </span>
        )}

        {/* Elapsed counter */}
        {isTraining && (
          <span className="text-xs text-primary font-mono">{elapsed}s</span>
        )}
      </div>

      <p className="text-sm text-muted-foreground">{def.description}</p>

      {/* Advanced settings */}
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger asChild>
          <button className="flex w-full items-center justify-between text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2 rounded">
            Advanced settings
            <ChevronDown
              className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`}
            />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-3">
          {def.type === "decision_tree" && (
            <>
              <SliderRow
                label="Max Depth"
                value={dtParams.max_depth}
                min={2}
                max={20}
                onChange={(v) => setDtParams({ ...dtParams, max_depth: v })}
              />
              <SliderRow
                label="Min Samples Split"
                value={dtParams.min_samples_split}
                min={2}
                max={20}
                onChange={(v) => setDtParams({ ...dtParams, min_samples_split: v })}
              />
            </>
          )}
          {def.type === "random_forest" && (
            <>
              <SliderRow
                label="N Estimators"
                value={rfParams.n_estimators}
                min={10}
                max={200}
                onChange={(v) => setRfParams({ ...rfParams, n_estimators: v })}
              />
              <SliderRow
                label="Max Depth"
                value={rfParams.max_depth}
                min={2}
                max={20}
                onChange={(v) => setRfParams({ ...rfParams, max_depth: v })}
              />
            </>
          )}
          {def.type === "xgboost" && (
            <>
              <SliderRow
                label="N Estimators"
                value={xgbParams.n_estimators}
                min={10}
                max={200}
                onChange={(v) => setXgbParams({ ...xgbParams, n_estimators: v })}
              />
              <SliderRow
                label="Learning Rate"
                value={xgbParams.learning_rate}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={(v) => setXgbParams({ ...xgbParams, learning_rate: v })}
                format={(v) => v.toFixed(2)}
              />
              <SliderRow
                label="Max Depth"
                value={xgbParams.max_depth}
                min={2}
                max={10}
                onChange={(v) => setXgbParams({ ...xgbParams, max_depth: v })}
              />
            </>
          )}
        </CollapsibleContent>
      </Collapsible>

      {/* Train button */}
      <Button
        onClick={onTrain}
        disabled={disabled}
        variant="hero"
        size="sm"
        className="w-full"
      >
        {isTraining ? (
          <>Training… {elapsed}s</>
        ) : (
          <>
            <Play className="h-4 w-4" />
            Train
          </>
        )}
      </Button>
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function Train() {
  const queryClient = useQueryClient();

  const [selectedImportId, setSelectedImportId] = useState<string>("");
  const [dtParams, setDtParams] = useState<DTParams>(DEFAULT_DT);
  const [rfParams, setRfParams] = useState<RFParams>(DEFAULT_RF);
  const [xgbParams, setXgbParams] = useState<XGBParams>(DEFAULT_XGB);

  // Per-model state
  const [trainingModel, setTrainingModel] = useState<ModelType | null>(null);
  const [elapsed, setElapsed] = useState<Record<ModelType, number>>({
    decision_tree: 0,
    random_forest: 0,
    xgboost: 0,
  });
  const [results, setResults] = useState<Record<ModelType, ModelResult | null>>({
    decision_tree: null,
    random_forest: null,
    xgboost: null,
  });

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { data: imports = [], isLoading: importsLoading } = useQuery({
    queryKey: ["imports"],
    queryFn: getImports,
  });

  // Auto-select first import
  useEffect(() => {
    if (!selectedImportId && imports.length > 0) {
      setSelectedImportId(String(imports[0].id));
    }
  }, [imports, selectedImportId]);

  // Elapsed timer
  useEffect(() => {
    if (trainingModel) {
      timerRef.current = setInterval(() => {
        setElapsed((prev) => ({
          ...prev,
          [trainingModel]: (prev[trainingModel] ?? 0) + 1,
        }));
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [trainingModel]);

  const mutation = useMutation({
    mutationFn: trainModel,
    onSuccess: (data: TrainResponse, variables) => {
      const mt = variables.model_type as ModelType;
      setResults((prev) => ({
        ...prev,
        [mt]: { accuracy: data.metrics.accuracy, f1_score: data.metrics.f1_score },
      }));
      setTrainingModel(null);
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["results"] });
      toast.success(
        `${mt.replace(/_/g, " ")} trained — ${(data.metrics.accuracy * 100).toFixed(1)}% accuracy`
      );
    },
    onError: (_err, variables) => {
      const mt = variables.model_type as ModelType;
      setTrainingModel(null);
      setElapsed((prev) => ({ ...prev, [mt]: 0 }));
      toast.error(`Training failed for ${mt.replace(/_/g, " ")}`);
    },
  });

  const trainSingleModel = (modelType: ModelType) => {
    if (!selectedImportId) {
      toast.error("Select an import session first");
      return;
    }

    const params: Record<string, unknown> =
      modelType === "decision_tree"
        ? dtParams
        : modelType === "random_forest"
        ? rfParams
        : xgbParams;

    setTrainingModel(modelType);
    setElapsed((prev) => ({ ...prev, [modelType]: 0 }));

    mutation.mutate({
      import_id: Number(selectedImportId),
      model_type: modelType,
      params,
    });
  };

  const handleTrainAll = async () => {
    if (!selectedImportId) {
      toast.error("Select an import session first");
      return;
    }
    for (const def of MODEL_DEFS) {
      await new Promise<void>((resolve) => {
        trainSingleModel(def.type);
        // poll until training finishes for this model
        const interval = setInterval(() => {
          if (trainingModel !== def.type) {
            clearInterval(interval);
            resolve();
          }
        }, 500);
      });
    }
  };

  const isAnyTraining = trainingModel !== null;

  return (
    <div className="min-h-screen">

      <main className="container mx-auto px-4 pt-24 pb-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div>
            <h1 className="text-3xl font-bold md:text-4xl">Train Models</h1>
            <p className="mt-2 text-muted-foreground">
              Configure and train machine learning models on your imported data
            </p>
          </div>
          <Button
            onClick={handleTrainAll}
            disabled={isAnyTraining || !selectedImportId}
            variant="hero"
            className="shrink-0 gap-2"
          >
            <Play className="h-4 w-4" />
            Train All
          </Button>
        </motion.div>

        {/* Import Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8"
        >
          <GlassCard className="space-y-4">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Select Dataset</h2>
            </div>

            {importsLoading ? (
              <Skeleton className="h-10 w-full max-w-sm" />
            ) : imports.length === 0 ? (
              <EmptyState message="No import sessions yet. Go to Import to upload data first." />
            ) : (
              <Select value={selectedImportId} onValueChange={setSelectedImportId}>
                <SelectTrigger className="max-w-sm bg-muted/30 border-border/50">
                  <SelectValue placeholder="Choose an import session…" />
                </SelectTrigger>
                <SelectContent>
                  {imports.map((imp) => (
                    <SelectItem key={imp.id} value={String(imp.id)}>
                      {imp.filename}
                      {imp.record_count != null && (
                        <span className="ml-2 text-muted-foreground text-xs">
                          ({imp.record_count.toLocaleString()} rows)
                        </span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </GlassCard>
        </motion.div>

        {/* Model Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-8"
        >
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Models</h2>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {MODEL_DEFS.map((def) => (
              <ModelTrainCard
                key={def.type}
                def={def}
                isTraining={trainingModel === def.type}
                elapsed={elapsed[def.type]}
                result={results[def.type]}
                disabled={isAnyTraining || !selectedImportId}
                onTrain={() => trainSingleModel(def.type)}
                dtParams={dtParams}
                rfParams={rfParams}
                xgbParams={xgbParams}
                setDtParams={setDtParams}
                setRfParams={setRfParams}
                setXgbParams={setXgbParams}
              />
            ))}

            {/* Neural Network — Coming Soon */}
            <div className="relative rounded-2xl border border-white/5 bg-white/[0.02] p-5 space-y-4 opacity-50">
              <div className="absolute top-3 right-3">
                <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                  Coming Soon
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/5">
                  <Brain className="h-5 w-5 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-semibold text-muted-foreground">Neural Network</p>
                  <span className="text-xs text-muted-foreground bg-white/5 px-2 py-0.5 rounded-full">
                    ~74% typical
                  </span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground">
                Deep learning for complex pattern recognition. Under development.
              </p>
              <Button disabled size="sm" className="w-full" variant="outline">
                Train
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Training status banner */}
        <AnimatePresence>
          {isAnyTraining && trainingModel && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 12 }}
              className="mt-4 rounded-xl border border-primary/30 bg-primary/10 px-6 py-4 flex items-center gap-3"
            >
              <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
              <p className="text-sm text-primary">
                Training{" "}
                <span className="font-semibold capitalize">
                  {trainingModel.replace(/_/g, " ")}
                </span>
                … {elapsed[trainingModel]}s elapsed. All buttons are disabled until complete.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

    </div>
  );
}

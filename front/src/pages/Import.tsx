import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  FileSpreadsheet,
  Info,
  Trash2,
} from "lucide-react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { DropZone } from "@/components/shared/DropZone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { GlassCard } from "@/components/ui/GlassCard";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { getImports, deleteImport } from "@/lib/api";
import type { UploadResponse } from "@/lib/api";

// Mirror lib/api.ts so the upload XHR honours the configured backend
// (Docker/dev override) instead of a hardcoded port.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ── Column type detection ─────────────────────────────────────────────────

function detectColType(values: unknown[]): "number" | "date" | "string" {
  const samples = values
    .filter((v) => v !== null && v !== undefined && v !== "")
    .slice(0, 20);
  if (samples.length === 0) return "string";

  const numericCount = samples.filter((v) => !isNaN(Number(v))).length;
  if (numericCount / samples.length > 0.8) return "number";

  const dateCount = samples.filter((v) => {
    const d = new Date(String(v));
    return !isNaN(d.getTime()) && String(v).length >= 6;
  }).length;
  if (dateCount / samples.length > 0.8) return "date";

  return "string";
}

const TYPE_BADGE: Record<string, string> = {
  number: "bg-blue-500/20 text-blue-300",
  date: "bg-purple-500/20 text-purple-300",
  string: "bg-slate-500/20 text-slate-300",
};

// ── Preview Table ─────────────────────────────────────────────────────────

function PreviewTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return null;
  const columns = Object.keys(rows[0]);
  const colTypes: Record<string, "number" | "date" | "string"> = {};
  for (const col of columns) {
    colTypes[col] = detectColType(rows.map((r) => r[col]));
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10 bg-white/5">
            {columns.map((col) => (
              <th key={col} className="whitespace-nowrap px-4 py-3 text-left font-medium">
                <div className="flex flex-col gap-1">
                  <span className="text-foreground">{col}</span>
                  <span
                    className={`w-fit rounded-full px-2 py-0.5 text-xs font-medium ${TYPE_BADGE[colTypes[col]]}`}
                  >
                    {colTypes[col]}
                  </span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {rows.map((row, i) => (
            <tr key={i} className="transition-colors hover:bg-white/5">
              {columns.map((col) => (
                <td
                  key={col}
                  className="max-w-[200px] truncate whitespace-nowrap px-4 py-2 text-muted-foreground"
                >
                  {String(row[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function Import() {
  const queryClient = useQueryClient();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [uploadErrors, setUploadErrors] = useState<string[]>([]);
  const [uploadWarnings, setUploadWarnings] = useState<string[]>([]);

  const xhrRef = useRef<XMLHttpRequest | null>(null);

  const { data: imports = [], isLoading: importsLoading } = useQuery({
    queryKey: ["imports"],
    queryFn: getImports,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteImport,
    onSuccess: (res) => {
      toast.success(
        `Deleted import #${res.import_id} — ${res.games_deleted.toLocaleString()} games, ` +
          `${res.models_deleted} model${res.models_deleted === 1 ? "" : "s"} removed.`,
      );
      // Imports, models, results and the leaderboard all change.
      queryClient.invalidateQueries({ queryKey: ["imports"] });
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["leaderboard"] });
      queryClient.invalidateQueries({ queryKey: ["results"] });
    },
    onError: () => toast.error("Could not delete import."),
  });

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setUploadResult(null);
    setUploadErrors([]);
    setUploadWarnings([]);
    setUploadProgress(0);
  };

  const handleImport = () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadResult(null);
    setUploadErrors([]);
    setUploadWarnings([]);

    const formData = new FormData();
    formData.append("file", selectedFile);

    const xhr = new XMLHttpRequest();
    xhrRef.current = xhr;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      setIsUploading(false);
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          setUploadResult(body as UploadResponse);
          setUploadWarnings(body.warnings ?? []);
          toast.success(`Imported ${body.row_count.toLocaleString()} rows successfully`);
          setSelectedFile(null);
          setDescription("");
          setUploadProgress(0);
          queryClient.invalidateQueries({ queryKey: ["imports"] });
        } else {
          const detail = body?.detail;
          if (detail && typeof detail === "object") {
            setUploadErrors(detail.errors ?? []);
            setUploadWarnings(detail.warnings ?? []);
          } else {
            setUploadErrors([typeof detail === "string" ? detail : "Upload failed."]);
          }
          toast.error("Import failed — see validation panel");
        }
      } catch {
        setUploadErrors(["Unexpected server response."]);
        toast.error("Import failed");
      }
    };

    xhr.onerror = () => {
      setIsUploading(false);
      setUploadErrors(["Network error. Is the backend running?"]);
      toast.error("Network error");
    };

    xhr.open("POST", `${BASE_URL}/api/imports/upload`);
    xhr.send(formData);
  };

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
          <h1 className="text-3xl font-bold md:text-4xl">Import Data</h1>
          <p className="mt-2 text-muted-foreground">
            Upload historical NBA game data for training and predictions
          </p>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Upload section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="space-y-6 lg:col-span-2"
          >
            <GlassCard className="space-y-6">
              <h2 className="text-xl font-semibold">Upload File</h2>

              <DropZone onFileSelect={handleFileSelect} />

              <div className="space-y-2">
                <label className="text-sm font-medium">Description (optional)</label>
                <Input
                  placeholder="e.g., Regular season 2024 data"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="border-border/50 bg-muted/30"
                  disabled={isUploading}
                />
              </div>

              {/* Progress bar */}
              <AnimatePresence>
                {isUploading && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-2"
                  >
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Uploading…</span>
                      <span className="font-medium text-primary">{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                  </motion.div>
                )}
              </AnimatePresence>

              <Button
                onClick={handleImport}
                disabled={!selectedFile || isUploading}
                variant="hero"
                size="lg"
                className="w-full focus-visible:ring-orange-500"
              >
                {isUploading ? "Uploading…" : "Import Data"}
              </Button>
            </GlassCard>

            {/* Validation panel */}
            <AnimatePresence>
              {(uploadErrors.length > 0 || uploadWarnings.length > 0 || uploadResult) && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 12 }}
                  className="space-y-3"
                >
                  {uploadErrors.map((err, i) => (
                    <div
                      key={i}
                      className="flex gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4"
                    >
                      <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
                      <p className="text-sm text-red-300">{err}</p>
                    </div>
                  ))}

                  {uploadWarnings.map((warn, i) => (
                    <div
                      key={i}
                      className="flex gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4"
                    >
                      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
                      <p className="text-sm text-amber-300">{warn}</p>
                    </div>
                  ))}

                  {uploadResult && (
                    <div className="flex gap-3 rounded-xl border border-green-500/30 bg-green-500/10 p-4">
                      <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-green-400" />
                      <p className="text-sm text-green-300">
                        Successfully imported{" "}
                        <span className="font-semibold">
                          {uploadResult.row_count.toLocaleString()} rows
                        </span>
                        {uploadResult.warnings?.length > 0 && (
                          <span className="ml-1 text-amber-300">
                            with {uploadResult.warnings.length} warning
                            {uploadResult.warnings.length > 1 ? "s" : ""}
                          </span>
                        )}
                      </p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Preview table */}
            <AnimatePresence>
              {uploadResult?.preview && uploadResult.preview.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <GlassCard className="space-y-4">
                    <h3 className="font-semibold">
                      Data Preview{" "}
                      <span className="ml-2 text-sm font-normal text-muted-foreground">
                        (first {uploadResult.preview.length} rows)
                      </span>
                    </h3>
                    <PreviewTable rows={uploadResult.preview} />
                  </GlassCard>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Requirements panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <GlassCard className="space-y-4">
              <div className="flex items-center gap-2">
                <Info className="h-5 w-5 text-secondary" />
                <h3 className="font-semibold">File Requirements</h3>
              </div>

              <div className="space-y-3 text-sm">
                {[
                  { label: "Formats", value: "CSV, XLSX, XLS" },
                  { label: "Required columns", value: "home_team, away_team, game_date" },
                  {
                    label: "Optional",
                    value: "home_score, away_score (needed for training)",
                  },
                  { label: "Max size", value: "50 MB" },
                ].map(({ label, value }) => (
                  <div key={label} className="flex items-start gap-3">
                    <div className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                    <p className="text-muted-foreground">
                      <span className="text-foreground">{label}:</span> {value}
                    </p>
                  </div>
                ))}
              </div>
            </GlassCard>
          </motion.div>
        </div>

        {/* Import history */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-12"
        >
          <h2 className="mb-6 text-xl font-semibold">Import History</h2>

          <GlassCard className="overflow-hidden p-0">
            {importsLoading ? (
              <div className="divide-y divide-border/50">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex gap-4 px-6 py-4">
                    <Skeleton className="h-5 w-5" />
                    <Skeleton className="h-5 flex-1" />
                    <Skeleton className="h-5 w-24" />
                  </div>
                ))}
              </div>
            ) : imports.length === 0 ? (
              <div className="p-6">
                <EmptyState
                  message="No imports yet. Upload your first dataset above."
                  icon={FileSpreadsheet}
                />
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border/50 bg-muted/30">
                      <th className="px-6 py-4 text-left text-sm font-semibold">Filename</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold">Rows</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold">Status</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold">Imported</th>
                      <th className="px-6 py-4 text-right text-sm font-semibold">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {imports.map((item, index) => (
                      <motion.tr
                        key={item.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: 0.4 + index * 0.05 }}
                        className="transition-colors hover:bg-muted/30"
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <FileSpreadsheet className="h-5 w-5 text-muted-foreground" />
                            <span className="font-medium">{item.filename}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-muted-foreground">
                          {item.record_count != null
                            ? item.record_count.toLocaleString()
                            : "—"}
                        </td>
                        <td className="px-6 py-4">
                          <span
                            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              item.status === "completed"
                                ? "bg-green-500/20 text-green-300"
                                : "bg-amber-500/20 text-amber-300"
                            }`}
                          >
                            {item.status ?? "completed"}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-muted-foreground">
                          {item.import_date
                            ? new Date(item.import_date).toLocaleDateString()
                            : "—"}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                disabled={deleteMutation.isPending}
                                className="focus-visible:ring-orange-500"
                                aria-label={`Delete import ${item.filename}`}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                              <AlertDialogHeader>
                                <AlertDialogTitle>Delete this import?</AlertDialogTitle>
                                <AlertDialogDescription>
                                  This permanently removes <span className="font-medium text-foreground">{item.filename}</span>{" "}
                                  (#{item.id})
                                  {item.record_count != null && (
                                    <> — {item.record_count.toLocaleString()} games</>
                                  )}{" "}
                                  and any models trained on it. Saved user predictions are kept.
                                  This cannot be undone.
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel>Cancel</AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={() => deleteMutation.mutate(item.id)}
                                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                >
                                  Delete
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </GlassCard>
        </motion.div>
      </main>

    </div>
  );
}

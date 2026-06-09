// Configurable at build time via `VITE_API_BASE_URL` (used in Docker).
// Falls back to the local-dev backend port.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ── Error class ────────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

// ── Interfaces ─────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface ImportRecord {
  id: number;
  filename: string;
  import_date: string;
  file_path: string | null;
  record_count: number | null;
  description: string | null;
  status: string;
}

export interface UploadResponse {
  import_id: number;
  row_count: number;
  warnings: string[];
  preview: Record<string, unknown>[];
}

export interface ModelRecord {
  id: number;
  name: string;
  type: string;
  parameters: string | null;
  model_path?: string | null;
  created_date: string;
  // metrics (joined from prediction_results)
  accuracy?: number | null;
  precision_score?: number | null;
  recall?: number | null;
  f1_score?: number | null;
  confidence?: number | null;
}

export interface TrainRequest {
  import_id: number;
  model_type: string;
  params?: Record<string, unknown>;
}

export interface TrainMetrics {
  accuracy: number;
  precision_score: number;
  recall: number;
  f1_score: number;
  confidence: number;
  confidence_reliable: boolean;
}

export interface TrainResponse {
  model_id: number;
  model_type: string;
  metrics: TrainMetrics;
}

export interface ResultRecord {
  name: string;
  type: string;
  accuracy: number;
  precision_score: number;
  recall: number;
  f1_score: number;
  confidence: number;
  created_date: string;
  feature_importance?: string | null;
}

export interface PredictRequest {
  home_team: string;
  away_team: string;
  season: string;
  model_id?: number;
  model_type?: string;
}

export interface PredictResponse {
  prediction: string;
  confidence: number;
  confidence_reliable: boolean;
  model_name: string;
  feature_importance: Record<string, number> | null;
}

export interface LeaderboardEntry {
  rank: number;
  model_id: number;
  model_name: string | null;
  model_type: string | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  cv_accuracy_mean: number | null;
  cv_accuracy_std: number | null;
  confidence: number | null;
  calibration_method: string | null;
  evaluated_games: number;
  created_date: string | null;
  sorted_by: string;
}

export type LeaderboardSort = "accuracy" | "f1" | "cv";

export interface UserLeaderboardEntry {
  rank: number;
  user_handle: string;
  resolved_count: number;
  correct_count: number;
  accuracy: number | null;
}

export interface FreshnessStatus {
  configured: boolean;
  latest_import_id: number | null;
  latest_game_date: string | null;
  total_games: number;
}

export interface FreshnessSyncRequest {
  seasons: number[];
  import_id?: number;
  start_date?: string;
  retrain?: string[];
  max_games?: number;
}

export interface FreshnessSyncResult {
  fetched: number;
  inserted: number;
  import_id: number;
  resolved_predictions: number;
  retrained: { model_type: string; model_id?: number; accuracy?: number | null; error?: string }[];
}

export interface SavePredictionRequest {
  user_handle: string;
  home_team: string;
  away_team: string;
  season: string;
  model_id?: number;
  model_type?: string;
  game_date?: string;
}

export interface SavedPrediction {
  id: number;
  user_handle: string;
  home_team: string;
  away_team: string;
  season: string | null;
  predicted_label: string;        // 'home_win' | 'away_win'
  prediction: string;             // 'Home Win' | 'Away Win'
  predicted_confidence: number | null;
  model_id: number | null;
  model_name: string | null;
  game_date?: string | null;
  resolved?: number;
  actual_label?: string | null;
  correct?: number | null;
  created_at?: string;
  confidence_reliable?: boolean;
}

export interface VerifyPerModel {
  model_name: string;
  accuracy: number;
  correct: number;
  total: number;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
}

export interface VerifyResponse {
  total: number;
  correct: number;
  accuracy: number;
  per_model: VerifyPerModel[];
}

export interface ImportTeamsResponse {
  import_id: number;
  teams: number[];
}

// ── Internal fetch helper ──────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (body.detail && typeof body.detail === "object") {
        detail = body.detail.message ?? JSON.stringify(body.detail);
      }
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

// ── Public API functions ───────────────────────────────────────────────────

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function getImports(): Promise<ImportRecord[]> {
  return request<ImportRecord[]>("/api/imports");
}

export async function uploadCSV(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/api/imports/upload`, {
    method: "POST",
    body: formData,
    // No Content-Type header — browser sets it with boundary automatically
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (body.detail && typeof body.detail === "object") {
        detail = body.detail.message ?? JSON.stringify(body.detail);
      }
    } catch {
      // ignore
    }
    throw new ApiError(res.status, detail);
  }

  return res.json() as Promise<UploadResponse>;
}

export function getImportTeams(id: number): Promise<ImportTeamsResponse> {
  return request<ImportTeamsResponse>(`/api/imports/${id}/teams`);
}

export interface DeleteImportResult {
  import_id: number;
  games_deleted: number;
  results_deleted: number;
  models_deleted: number;
  model_files_removed: number;
}

export function deleteImport(id: number): Promise<DeleteImportResult> {
  return request<DeleteImportResult>(`/api/imports/${id}`, { method: "DELETE" });
}

export function getModels(): Promise<ModelRecord[]> {
  return request<ModelRecord[]>("/api/models");
}

export function trainModel(body: TrainRequest): Promise<TrainResponse> {
  return request<TrainResponse>("/api/train", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getResults(importId?: number): Promise<ResultRecord[]> {
  const qs = importId != null ? `?import_id=${importId}` : "";
  return request<ResultRecord[]>(`/api/results${qs}`);
}

export function predictGame(body: PredictRequest): Promise<PredictResponse> {
  return request<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function savePrediction(body: SavePredictionRequest): Promise<SavedPrediction> {
  return request<SavedPrediction>("/api/predictions", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getLeaderboard(
  sort: LeaderboardSort = "accuracy",
  limit?: number,
  importId?: number,
): Promise<LeaderboardEntry[]> {
  const params = new URLSearchParams({ sort });
  if (limit != null) params.set("limit", String(limit));
  if (importId != null) params.set("import_id", String(importId));
  return request<LeaderboardEntry[]>(`/api/leaderboard?${params.toString()}`);
}

export function getUserLeaderboard(user?: string): Promise<UserLeaderboardEntry[]> {
  const qs = user ? `?user=${encodeURIComponent(user)}` : "";
  return request<UserLeaderboardEntry[]>(`/api/user-leaderboard${qs}`);
}

export function getFreshnessStatus(): Promise<FreshnessStatus> {
  return request<FreshnessStatus>("/api/freshness/status");
}

export function syncFreshness(body: FreshnessSyncRequest): Promise<FreshnessSyncResult> {
  return request<FreshnessSyncResult>("/api/freshness/sync", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getPredictions(user?: string, limit = 50): Promise<SavedPrediction[]> {
  const params = new URLSearchParams();
  if (user) params.set("user", user);
  params.set("limit", String(limit));
  return request<SavedPrediction[]>(`/api/predictions?${params.toString()}`);
}

export function getVerification(id: number): Promise<VerifyResponse> {
  return request<VerifyResponse>(`/api/verify/${id}`);
}

"""
NBA Prediction System — FastAPI REST layer.
Run from the backend/ directory:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Minimal log config (level via NBA_LOG_LEVEL). Intentionally not a structured
# logging framework — uvicorn handles access logs; this covers app loggers.
logging.basicConfig(
    level=os.environ.get("NBA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from balldontlie_client import BalldontlieError
from data_freshness import FreshnessPipeline
from data_importer import DataImporter, validate_csv
from database import NBADatabase
from predictive_models import DB_TYPE, PredictiveModels

import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Sets KEY=VALUE pairs into the
    environment for any key not already set — real env vars always win. Used so
    a local BALLDONTLIE_API_KEY / NBA_API_KEY in the project-root .env is picked
    up in dev. In Docker/prod, pass real env vars instead."""
    if not path.is_file():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv(_PROJECT_ROOT / ".env")

DB_PATH = os.environ.get("NBA_DB_PATH", str(_PROJECT_ROOT / "data" / "nba_predictions.db"))
UPLOADS_DIR = Path(os.environ.get("NBA_UPLOADS_DIR", str(_PROJECT_ROOT / "data" / "uploads")))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Security: shared-secret API key auth
# ──────────────────────────────────────────────
# Set NBA_API_KEY in the env to require an X-API-Key header on mutating endpoints
# (upload, train, predict). Leave unset for local dev — no auth is enforced.
_API_KEY = os.environ.get("NBA_API_KEY", "").strip() or None


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if _API_KEY is None:
        return  # dev mode — auth disabled
    if not x_api_key or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


# ──────────────────────────────────────────────
# App, CORS, rate limiting
# ──────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="NBA Prediction API", version="1.0.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# Permissive in dev (single localhost origin). Override via NBA_CORS_ORIGINS=https://a,https://b in prod.
_origins = [o.strip() for o in os.environ.get("NBA_CORS_ORIGINS", "http://localhost:8080").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Shared instances (created once at startup)
# ──────────────────────────────────────────────
db = NBADatabase(db_path=DB_PATH)
importer = DataImporter(db=db)
pm = PredictiveModels(db=db)
freshness = FreshnessPipeline(db=db, importer=importer, pm=pm)

# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class TrainRequest(BaseModel):
    import_id: int
    model_type: str  # decision_tree | random_forest | xgboost | baseline
    params: Optional[Dict[str, Any]] = {}


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    season: str
    model_id: Optional[int] = None
    model_type: Optional[str] = None


class SavePredictionRequest(BaseModel):
    user_handle: str
    home_team: str
    away_team: str
    season: str
    model_id: Optional[int] = None
    model_type: Optional[str] = None
    game_date: Optional[str] = None


class FreshnessSyncRequest(BaseModel):
    seasons: List[int]                       # starting years, e.g. [2022, 2023, 2024, 2025]
    import_id: Optional[int] = None          # which dataset to extend (default: latest)
    start_date: Optional[str] = None         # YYYY-MM-DD, server-side incremental filter
    retrain: Optional[List[str]] = None      # model types to retrain after ingest
    max_games: Optional[int] = None          # cap fetched games (useful for a quick sync)


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Convert a pandas Series or dict-like to a plain dict with serialisable values."""
    d = dict(row)
    for k, v in d.items():
        if hasattr(v, 'isoformat'):
            d[k] = v.isoformat()
        elif isinstance(v, float) and (v != v):  # NaN
            d[k] = None
    return d


def _df_to_records(df: pd.DataFrame) -> List[dict]:
    return [_row_to_dict(r) for _, r in df.iterrows()]


def _num(v, ndigits: int = 4):
    """Coerce a possibly-NaN/None numpy scalar to a JSON-safe rounded float."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, ndigits)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Liveness + DB connectivity. Deliberately does NOT touch balldontlie —
    a health probe must not depend on (or burn the rate limit of) an external
    service. External-feed status lives at /api/freshness/status."""
    db_ok = db.ping()
    body = {
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(status_code=200 if db_ok else 503, content=body)


# ── Imports ──────────────────────────────────

@app.get("/api/imports")
def list_imports():
    df = db.get_import_records()
    if df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        r = _row_to_dict(row)
        r.setdefault("status", "completed")
        records.append(r)
    return records


@app.post("/api/imports/upload", dependencies=[Depends(require_api_key)])
@limiter.limit("20/hour")
async def upload_file(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(422, detail="Only CSV and XLSX files are supported.")

    # Save to uploads dir temporarily
    dest = UPLOADS_DIR / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Load for validation preview
    try:
        if ext == ".csv":
            df_raw = pd.read_csv(dest)
        else:
            df_raw = pd.read_excel(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, detail=f"Could not read file: {exc}")

    # Map column variations before validation
    from data_importer import DataImporter as _DI
    _tmp = _DI(db)
    df_mapped = _tmp._map_column_variations(df_raw)

    # Validate
    validation = validate_csv(df_mapped)
    if not validation["valid"]:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            422,
            detail={
                "message": "File failed validation",
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            },
        )

    # Full import
    try:
        import_id, df_clean = importer.import_historical_data(str(dest), description=file.filename)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, detail=str(exc))

    preview = df_mapped.head(10).fillna("").to_dict(orient="records")

    return {
        "import_id": import_id,
        "row_count": len(df_clean),
        "warnings": validation["warnings"],
        "preview": preview,
    }


@app.delete("/api/imports/{import_id}", dependencies=[Depends(require_api_key)])
@limiter.limit("30/hour")
def delete_import(request: Request, import_id: int):
    imports_df = db.get_import_records()
    if imports_df.empty or import_id not in imports_df["id"].values:
        raise HTTPException(404, detail=f"import_id {import_id} not found.")

    summary = db.delete_import(import_id)

    # Remove the trained-model files (.pkl + sidecar .json) off disk.
    files_removed = 0
    for path_str in summary.pop("model_paths", []):
        for f in (Path(path_str), Path(path_str).with_suffix(".json")):
            try:
                if f.exists():
                    f.unlink()
                    files_removed += 1
            except OSError:
                pass
    summary["model_files_removed"] = files_removed
    return summary


@app.get("/api/imports/{import_id}/teams")
def get_import_teams(import_id: int):
    teams = db.get_unique_teams(import_id)
    if not teams:
        raise HTTPException(404, detail=f"No game data found for import_id={import_id}")
    return {"import_id": import_id, "teams": teams}


@app.get("/api/imports/{import_id}/preview")
def get_import_preview(import_id: int):
    df = db.get_game_data(import_id)
    if df.empty:
        raise HTTPException(404, detail=f"No game data found for import_id={import_id}")
    return df.head(10).fillna("").to_dict(orient="records")


# ── Models ───────────────────────────────────

@app.get("/api/models")
def list_models():
    models_df = db.get_models()
    if models_df.empty:
        return []

    results_df = db.get_prediction_results()
    if results_df.empty:
        return _df_to_records(models_df)

    # Attach latest result metrics per model
    latest = (
        results_df.sort_values("created_date", ascending=False)
        .drop_duplicates(subset=["model_id"])
        [["model_id", "accuracy", "precision_score", "recall", "f1_score", "confidence"]]
    )
    merged = models_df.merge(latest, left_on="id", right_on="model_id", how="left")
    return _df_to_records(merged)


# ── Training ─────────────────────────────────

VALID_MODEL_TYPES = set(DB_TYPE.keys())


@app.post("/api/train", dependencies=[Depends(require_api_key)])
@limiter.limit("5/hour")
def train_model(request: Request, body: TrainRequest):
    # Validate import exists
    imports_df = db.get_import_records()
    if imports_df.empty or body.import_id not in imports_df["id"].values:
        raise HTTPException(400, detail=f"import_id {body.import_id} does not exist.")

    mt = body.model_type.lower()
    if mt not in VALID_MODEL_TYPES:
        raise HTTPException(
            400,
            detail=f"Invalid model_type '{body.model_type}'. "
                   f"Choose from: {sorted(VALID_MODEL_TYPES)}",
        )

    params = body.params or {}
    try:
        if mt == "decision_tree":
            model_id, metrics = pm.train_decision_tree(body.import_id, **params)
        elif mt == "random_forest":
            model_id, metrics = pm.train_random_forest(body.import_id, **params)
        elif mt == "xgboost":
            model_id, metrics = pm.train_xgboost(body.import_id, **params)
        else:  # baseline
            strategy = params.pop("strategy", "most_frequent")
            model_id, metrics = pm.train_baseline(body.import_id, strategy=strategy)
    except ImportError as exc:
        raise HTTPException(400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=f"Training failed: {exc}")

    return {
        "model_id": model_id,
        "model_type": mt,
        "metrics": metrics,
    }


# ── Results ───────────────────────────────────

@app.get("/api/results")
def get_results(import_id: Optional[int] = None):
    try:
        comparison = pm.get_model_comparison(import_record_id=import_id)
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))
    if comparison.empty:
        return []
    return _df_to_records(comparison)


# ── Leaderboard (model ranking) ───────────────

# Maps a `sort` query value to the prediction_results column to rank on.
_LEADERBOARD_SORT_COLUMNS = {
    "accuracy": "accuracy",
    "f1": "f1_score",
    "f1_score": "f1_score",
    "cv": "cv_accuracy_mean",
    "cv_accuracy": "cv_accuracy_mean",
    "cv_accuracy_mean": "cv_accuracy_mean",
}


@app.get("/api/leaderboard")
def leaderboard(sort: str = "accuracy", limit: Optional[int] = None,
                import_id: Optional[int] = None):
    """Rank trained models by a chosen metric using each model's latest
    prediction_results row. Default sort: test accuracy (desc).

    `import_id` scopes the board to models evaluated on one import — important
    once the freshness pipeline creates new imports, since accuracies measured
    on different test splits are not comparable."""
    results_df = db.get_prediction_results(import_record_id=import_id)
    models_df = db.get_models()
    if results_df.empty or models_df.empty:
        return []

    # One row per model — its most recent evaluation.
    latest = (
        results_df.sort_values("created_date", ascending=False)
        .drop_duplicates(subset=["model_id"])
    )
    merged = latest.merge(models_df, left_on="model_id", right_on="id", suffixes=("", "_model"))

    sort_col = _LEADERBOARD_SORT_COLUMNS.get(sort.lower(), "accuracy")
    if sort_col not in merged.columns:
        sort_col = "accuracy"
    merged = merged.sort_values(sort_col, ascending=False, na_position="last")

    rows: List[dict] = []
    for rank, (_, r) in enumerate(merged.iterrows(), start=1):
        try:
            preds = json.loads(r["predictions"]) if r.get("predictions") else []
        except (TypeError, json.JSONDecodeError):
            preds = []
        rows.append({
            "rank": rank,
            "model_id": int(r["model_id"]),
            "model_name": r.get("name"),
            "model_type": r.get("type"),
            "accuracy": _num(r.get("accuracy")),
            "precision": _num(r.get("precision_score")),
            "recall": _num(r.get("recall")),
            "f1_score": _num(r.get("f1_score")),
            "cv_accuracy_mean": _num(r.get("cv_accuracy_mean")),
            "cv_accuracy_std": _num(r.get("cv_accuracy_std")),
            "confidence": _num(r.get("confidence")),
            "calibration_method": r.get("calibration_method"),
            "evaluated_games": len(preds),
            "created_date": r.get("created_date"),
            "sorted_by": sort_col,
        })

    if limit:
        rows = rows[: int(limit)]
    return rows


# ── Predictions ───────────────────────────────

@app.post("/api/predict", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def predict_game(request: Request, body: PredictRequest):
    if body.model_id is None and body.model_type is None:
        raise HTTPException(400, detail="Provide model_id or model_type.")
    try:
        result = pm.predict_single(
            home_team=body.home_team,
            away_team=body.away_team,
            season=body.season,
            model_id=body.model_id,
            model_type=body.model_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=f"Prediction failed: {exc}")
    return result


# ── Saved predictions (handle-keyed) ──────────

def _prediction_record(row, model_name_map: Dict[int, str]) -> dict:
    r = _row_to_dict(row)
    r["model_name"] = model_name_map.get(r.get("model_id"))
    # Human-friendly display label alongside the stored snake_case label.
    r["prediction"] = "Home Win" if r.get("predicted_label") == "home_win" else "Away Win"
    return r


@app.post("/api/predictions", dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
def create_prediction(request: Request, body: SavePredictionRequest):
    handle = (body.user_handle or "").strip()
    if not handle:
        raise HTTPException(400, detail="user_handle is required.")
    if len(handle) > 40:
        raise HTTPException(400, detail="user_handle must be 40 characters or fewer.")
    if body.model_id is None and body.model_type is None:
        raise HTTPException(400, detail="Provide model_id or model_type.")

    # Compute the prediction server-side so the stored label/confidence are the
    # model's own output, not client-supplied (honor-system handles, trusted model).
    try:
        result = pm.predict_single(
            home_team=body.home_team,
            away_team=body.away_team,
            season=body.season,
            model_id=body.model_id,
            model_type=body.model_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=f"Prediction failed: {exc}")

    label = result["predicted_label"]
    pred_id = db.save_prediction(
        user_handle=handle,
        home_team=body.home_team,
        away_team=body.away_team,
        season=body.season,
        predicted_label=label,
        predicted_confidence=result["confidence"],
        model_id=result.get("model_id"),
        model_label=label,
        game_date=body.game_date,
    )

    return {
        "id": pred_id,
        "user_handle": handle,
        "home_team": body.home_team,
        "away_team": body.away_team,
        "season": body.season,
        "predicted_label": label,
        "prediction": result["prediction"],
        "predicted_confidence": result["confidence"],
        "model_id": result.get("model_id"),
        "model_name": result.get("model_name"),
        "confidence_reliable": result.get("confidence_reliable", True),
    }


@app.get("/api/predictions")
def list_predictions(user: Optional[str] = None, limit: Optional[int] = 50):
    df = db.get_predictions(user_handle=user, limit=limit)
    if df.empty:
        return []
    models_df = db.get_models()
    model_name_map = dict(zip(models_df["id"], models_df["name"])) if not models_df.empty else {}
    return [_prediction_record(row, model_name_map) for _, row in df.iterrows()]


@app.get("/api/user-leaderboard")
def user_leaderboard(user: Optional[str] = None):
    """Per-handle accuracy over *resolved* saved predictions. Seeds a future
    user leaderboard; the model leaderboard stays the primary board."""
    df = db.get_user_accuracy(user_handle=user)
    if df.empty:
        return []
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), start=1):
        rows.append({
            "rank": rank,
            "user_handle": r["user_handle"],
            "resolved_count": int(r["resolved_count"]),
            "correct_count": int(r["correct_count"]) if r["correct_count"] is not None else 0,
            "accuracy": _num(r["accuracy"]),
        })
    return rows


# ── Data freshness (balldontlie ingest + resolve + retrain) ──

@app.get("/api/freshness/status")
def freshness_status():
    """Whether the balldontlie key is configured and how current the data is."""
    latest_import = db.get_latest_import_id()
    latest_game_date = None
    total_games = 0
    if latest_import is not None:
        gd = db.get_game_data(latest_import)
        total_games = len(gd)
        if total_games and "game_date" in gd.columns:
            latest_game_date = str(gd["game_date"].max())
    return {
        "configured": freshness.configured,
        "latest_import_id": latest_import,
        "latest_game_date": latest_game_date,
        "total_games": total_games,
    }


# The sync endpoint is for SMALL incremental updates only. balldontlie's free
# tier is 5 req/min, so a multi-season backfill takes many minutes and would
# hang the worker past any HTTP timeout — that's what backfill_freshness.py is
# for. We cap the games an HTTP sync will fetch and require the caller to opt in.
FRESHNESS_SYNC_MAX_GAMES = 250


@app.post("/api/freshness/sync", dependencies=[Depends(require_api_key)])
@limiter.limit("5/hour")
def freshness_sync(request: Request, body: FreshnessSyncRequest):
    # Request-shape validation first (so these 400s are deterministic regardless
    # of whether a key is configured), then the server-readiness check.
    if not body.seasons:
        raise HTTPException(400, detail="Provide at least one season (starting year).")
    if body.max_games is None or body.max_games > FRESHNESS_SYNC_MAX_GAMES:
        raise HTTPException(
            400,
            detail=(
                f"This endpoint is for small incremental syncs only — set max_games "
                f"≤ {FRESHNESS_SYNC_MAX_GAMES}. For a full backfill run the CLI: "
                f"`python3 backfill_freshness.py --seasons … --retrain …`."
            ),
        )
    if not freshness.configured:
        raise HTTPException(
            400,
            detail="balldontlie is not configured. Set BALLDONTLIE_API_KEY in the environment.",
        )
    if body.retrain:
        bad = [m for m in body.retrain if m.lower() not in VALID_MODEL_TYPES]
        if bad:
            raise HTTPException(400, detail=f"Invalid retrain model types: {bad}")
    try:
        result = freshness.sync(
            seasons=body.seasons,
            target_import_id=body.import_id,
            start_date=body.start_date,
            retrain=body.retrain,
            max_games=body.max_games,
        )
    except BalldontlieError as exc:
        raise HTTPException(502, detail=f"balldontlie error: {exc}")
    except Exception as exc:
        raise HTTPException(500, detail=f"Sync failed: {exc}")
    return result


# ── Verification ──────────────────────────────

@app.get("/api/verify/{import_id}")
def verify_import(import_id: int):
    imports_df = db.get_import_records()
    if imports_df.empty or import_id not in imports_df["id"].values:
        raise HTTPException(404, detail=f"import_id {import_id} not found.")

    results_df = db.get_prediction_results(import_record_id=import_id)
    if results_df.empty:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "per_model": []}

    models_df = db.get_models()
    model_name_map = dict(zip(models_df["id"], models_df["name"])) if not models_df.empty else {}

    per_model = []
    total_correct = 0
    total_games = 0

    for _, row in results_df.iterrows():
        try:
            preds = json.loads(row["predictions"]) if row["predictions"] else []
            actuals = json.loads(row["actual_results"]) if row["actual_results"] else []
        except (TypeError, json.JSONDecodeError):
            preds, actuals = [], []

        n = min(len(preds), len(actuals))
        if n == 0:
            continue

        # Confusion matrix (LabelEncoder convention: 1 = home_win, 0 = away_win)
        tp = sum(1 for p, a in zip(preds[:n], actuals[:n]) if p == 1 and a == 1)
        tn = sum(1 for p, a in zip(preds[:n], actuals[:n]) if p == 0 and a == 0)
        fp = sum(1 for p, a in zip(preds[:n], actuals[:n]) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(preds[:n], actuals[:n]) if p == 0 and a == 1)
        correct = tp + tn
        total_correct += correct
        total_games += n

        per_model.append({
            "model_name": model_name_map.get(row["model_id"], f"Model {row['model_id']}"),
            "accuracy": round(correct / n, 4),
            "correct": correct,
            "total": n,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })

    return {
        "total": total_games,
        "correct": total_correct,
        "accuracy": round(total_correct / total_games, 4) if total_games else 0.0,
        "per_model": per_model,
    }


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

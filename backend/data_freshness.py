"""
Data-freshness pipeline: pull recent finals from balldontlie, fold them into the
existing wide game_data table, resolve users' saved predictions against the new
finals, and optionally retrain + promote a model.

Flow:  fetch (balldontlie) → map to wide rows → clean (DataImporter) →
       dedup-append (NBADatabase) → resolve predictions → optional retrain.

Tier note: a free key yields scores only. The 10 box-score features mean-fallback
for these new games; the 24 score/schedule features (Elo, rest, travel, venue,
h2h, calendar) stay fresh. Upgrading to a GOAT key enables box-score enrichment
with no change here (see balldontlie_client.iter_box_scores).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from balldontlie_client import BalldontlieClient
from data_importer import DataImporter
from team_locations import abbr_to_team_id

logger = logging.getLogger(__name__)


class FreshnessPipeline:
    def __init__(self, db, importer: Optional[DataImporter] = None,
                 client: Optional[BalldontlieClient] = None, pm=None):
        self.db = db
        self.importer = importer or DataImporter(db)
        self.client = client or BalldontlieClient()
        self.pm = pm

    @property
    def configured(self) -> bool:
        return self.client.configured

    # ------------------------------------------------------------------ #
    # Mapping balldontlie → wide importer rows                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def map_game(game: Dict) -> Optional[Dict]:
        """Translate one balldontlie game into a wide row keyed on NBA Stats IDs.
        Returns None for unmappable rows (defunct team, missing score)."""
        home = game.get("home_team") or {}
        away = game.get("visitor_team") or {}
        home_id = abbr_to_team_id(home.get("abbreviation"))
        away_id = abbr_to_team_id(away.get("abbreviation"))
        if not home_id or not away_id:
            return None
        hs, as_ = game.get("home_team_score"), game.get("visitor_team_score")
        if hs is None or as_ is None:
            return None
        # Synthetic GAME_ID whose leading digit drives season_type in
        # DataImporter._clean_data (4 = playoffs, 2 = regular season).
        lead = "4" if game.get("postseason") else "2"
        return {
            "home_team": home_id,
            "away_team": away_id,
            "home_score": int(hs),
            "away_score": int(as_),
            "game_date": game.get("date"),
            "season": str(game.get("season")),
            "GAME_ID": f"{lead}{game.get('id')}",
        }

    def fetch_wide(self, seasons: List[int], start_date: Optional[str] = None,
                   max_games: Optional[int] = None) -> pd.DataFrame:
        rows = []
        for game in self.client.iter_finals(seasons, start_date=start_date, max_games=max_games):
            mapped = self.map_game(game)
            if mapped:
                rows.append(mapped)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Ingest + sync                                                       #
    # ------------------------------------------------------------------ #
    def ingest(self, seasons: List[int], target_import_id: Optional[int] = None,
               start_date: Optional[str] = None, max_games: Optional[int] = None) -> Dict:
        """Fetch finals and dedup-append them to the target import. Creates a
        'live' import if none exists yet."""
        df = self.fetch_wide(seasons, start_date=start_date, max_games=max_games)
        target = target_import_id or self.db.get_latest_import_id()
        if target is None:
            target = self.db.register_import(
                "balldontlie_live", "balldontlie", 0, "Live data via balldontlie"
            )
        if df.empty:
            return {"fetched": 0, "inserted": 0, "import_id": target}

        # Reuse the importer's column mapping + cleaning (derives result + season_type).
        clean = self.importer._clean_data(self.importer._map_column_variations(df))
        clean = clean.copy()
        clean["game_date"] = (
            pd.to_datetime(clean["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        )
        inserted = self.db.append_game_data(target, clean)
        return {"fetched": len(df), "inserted": inserted, "import_id": target}

    def sync(self, seasons: List[int], target_import_id: Optional[int] = None,
             start_date: Optional[str] = None, retrain: Optional[List[str]] = None,
             max_games: Optional[int] = None) -> Dict:
        """Full freshness cycle: ingest → resolve saved predictions → optional
        retrain + promote (promotion is implicit — predict/leaderboard always use
        the latest model of each type)."""
        result = self.ingest(seasons, target_import_id, start_date, max_games)
        result["resolved_predictions"] = self.db.resolve_predictions()

        result["retrained"] = []
        if retrain and self.pm and result["inserted"] > 0:
            for mt in retrain:
                try:
                    model_id, metrics = self._train(mt, result["import_id"])
                    result["retrained"].append({
                        "model_type": mt, "model_id": model_id,
                        "accuracy": metrics.get("accuracy"),
                        "cv_accuracy_mean": metrics.get("cv_accuracy_mean"),
                    })
                except Exception as exc:  # don't fail the whole sync on one model
                    logger.warning("retrain %s failed: %s", mt, exc)
                    result["retrained"].append({"model_type": mt, "error": str(exc)})
        return result

    def _train(self, model_type: str, import_id: int):
        mt = model_type.lower()
        if mt == "random_forest":
            return self.pm.train_random_forest(import_id)
        if mt == "decision_tree":
            return self.pm.train_decision_tree(import_id)
        if mt == "xgboost":
            return self.pm.train_xgboost(import_id)
        if mt == "baseline":
            return self.pm.train_baseline(import_id)
        raise ValueError(f"unknown model_type: {model_type}")

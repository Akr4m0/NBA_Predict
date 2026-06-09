"""
Thin balldontlie NBA API client for the data-freshness pipeline.

Free tier (`/nba/v1/games`) returns final scores, dates, teams and season —
but NOT box-score stats (FG%, AST, REB …); those are GOAT-tier only. The client
is therefore *tier-agnostic*: it always ingests games, and box-score enrichment
(`iter_box_scores`) simply yields nothing / degrades gracefully on a free key,
so upgrading to a GOAT key later starts pulling box scores with no code change.

Auth: reads the key from `BALLDONTLIE_API_KEY` (or pass `api_key=`).
Rate limiting: free tier is 5 req/min → default `min_interval=12s` between calls.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterator, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.balldontlie.io/nba/v1"


class BalldontlieError(RuntimeError):
    """Raised for configuration or unrecoverable API errors."""


class BalldontlieClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        min_interval: float = 13.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_rate_limit_retries: int = 8,
    ):
        self.api_key = (api_key or os.environ.get("BALLDONTLIE_API_KEY", "")).strip()
        # Seconds between requests. Free tier is 5 req/min; 13s (~4.6/min) leaves
        # margin so we don't trip the rolling-window limit on a long backfill.
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries                    # network/transport errors
        self.max_rate_limit_retries = max_rate_limit_retries  # 429s (escalating backoff)
        self._last_request = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------ #
    # Low-level request with throttle + 429 backoff                       #
    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _get(self, path: str, params: Dict) -> Dict:
        if not self.api_key:
            raise BalldontlieError(
                "BALLDONTLIE_API_KEY is not set — cannot reach balldontlie."
            )
        headers = {"Authorization": self.api_key}
        network_retries = 0
        rate_limit_retries = 0
        while True:
            self._throttle()
            try:
                resp = httpx.get(
                    f"{BASE_URL}{path}", params=params, headers=headers, timeout=self.timeout
                )
            except httpx.HTTPError as exc:  # network/timeout
                self._last_request = time.monotonic()
                network_retries += 1
                if network_retries > self.max_retries:
                    raise BalldontlieError(f"balldontlie request failed: {exc}")
                logger.warning("balldontlie network error (retry %d): %s", network_retries, exc)
                time.sleep(min(5.0 * network_retries, 15.0))
                continue
            self._last_request = time.monotonic()

            if resp.status_code == 429:  # rate limited — escalating backoff
                rate_limit_retries += 1
                if rate_limit_retries > self.max_rate_limit_retries:
                    raise BalldontlieError(
                        "balldontlie rate limit: retries exhausted. The free tier is "
                        "5 req/min — wait a minute and re-run."
                    )
                wait = min(20.0 * rate_limit_retries, 75.0)  # 20,40,60,75,75…
                logger.warning("balldontlie 429; backing off %.0fs (retry %d/%d)",
                               wait, rate_limit_retries, self.max_rate_limit_retries)
                time.sleep(wait)
                continue
            if resp.status_code in (401, 403):
                raise BalldontlieError(
                    f"balldontlie auth/tier error {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------ #
    # Public: iterate finished games                                      #
    # ------------------------------------------------------------------ #
    def iter_finals(
        self,
        seasons: List[int],
        start_date: Optional[str] = None,
        per_page: int = 100,
        max_games: Optional[int] = None,
    ) -> Iterator[Dict]:
        """Yield finished ('Final') games for the given seasons, cursor-paginated.

        `seasons` are starting years (2024 = the 2024-25 season).
        `start_date` (YYYY-MM-DD) filters server-side for incremental syncs.
        """
        cursor: Optional[int] = None
        yielded = 0
        while True:
            params: Dict = {"seasons[]": seasons, "per_page": per_page}
            if start_date:
                params["start_date"] = start_date
            if cursor is not None:
                params["cursor"] = cursor

            payload = self._get("/games", params)
            for game in payload.get("data", []):
                # Only completed games carry a usable outcome.
                if str(game.get("status", "")).strip().lower() != "final":
                    continue
                yield game
                yielded += 1
                if max_games is not None and yielded >= max_games:
                    return

            cursor = payload.get("meta", {}).get("next_cursor")
            if not cursor:
                return

    def iter_box_scores(self, date: str) -> Iterator[Dict]:
        """Yield per-team box scores for a date (GOAT tier). On a free key this
        raises BalldontlieError(401/403); callers should catch and skip so the
        pipeline degrades to scores-only. Wired now so a GOAT upgrade is a
        config change, not a code change."""
        payload = self._get("/box_scores", {"date": date})
        for entry in payload.get("data", []):
            yield entry

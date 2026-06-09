"""FastAPI endpoint smoke tests via TestClient."""


def test_health(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] is True  # DB connectivity reported
    assert "timestamp" in body


def test_imports_empty_on_fresh_db(api_client):
    r = api_client.get("/api/imports")
    assert r.status_code == 200
    assert r.json() == []


def test_models_empty_on_fresh_db(api_client):
    r = api_client.get("/api/models")
    assert r.status_code == 200
    assert r.json() == []


def test_predict_requires_model_id_or_type(api_client):
    r = api_client.post("/api/predict", json={
        "home_team": "1610612737",
        "away_team": "1610612738",
        "season": "2019-20",
    })
    assert r.status_code == 400
    assert "model_id" in r.json()["detail"]


def test_train_rejects_unknown_import(api_client):
    r = api_client.post("/api/train", json={
        "import_id": 9999,
        "model_type": "random_forest",
        "params": {},
    })
    assert r.status_code == 400


def test_train_rejects_invalid_model_type(api_client, tmp_path):
    # Need a real import so we get past the import-id check.
    from database import NBADatabase
    import os
    db = NBADatabase(db_path=os.environ["NBA_DB_PATH"])
    iid = db.register_import("x.csv", "/tmp/x.csv", record_count=0)

    r = api_client.post("/api/train", json={
        "import_id": iid,
        "model_type": "nonsense",
        "params": {},
    })
    assert r.status_code == 400
    assert "Invalid model_type" in r.json()["detail"]


def test_upload_rejects_bad_extension(api_client):
    r = api_client.post(
        "/api/imports/upload",
        files={"file": ("not_a_csv.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 422


def test_verify_unknown_import_returns_404(api_client):
    r = api_client.get("/api/verify/9999")
    assert r.status_code == 404


# ── Auth (require_api_key dependency) ───────────────────────────────────────

def test_predict_without_api_key_is_401_when_key_required(api_client_with_key):
    r = api_client_with_key.post("/api/predict", json={
        "home_team": "1610612737", "away_team": "1610612738",
        "season": "2019-20", "model_type": "random_forest",
    })
    assert r.status_code == 401


def test_predict_with_wrong_api_key_is_401(api_client_with_key):
    r = api_client_with_key.post(
        "/api/predict",
        headers={"X-API-Key": "wrong"},
        json={"home_team": "1610612737", "away_team": "1610612738",
              "season": "2019-20", "model_type": "random_forest"},
    )
    assert r.status_code == 401


def test_predict_with_correct_api_key_passes_auth(api_client_with_key):
    # Auth succeeds; the call still fails (404) because there's no trained model
    # in the tmp DB. We just want a non-401.
    r = api_client_with_key.post(
        "/api/predict",
        headers={"X-API-Key": "secret"},
        json={"home_team": "1610612737", "away_team": "1610612738",
              "season": "2019-20", "model_type": "random_forest"},
    )
    assert r.status_code != 401, f"unexpected status {r.status_code}: {r.text}"


def test_read_endpoints_do_not_require_api_key(api_client_with_key):
    # /api/health, /api/imports, /api/models are reads — they should be open
    # even when NBA_API_KEY is enforced.
    assert api_client_with_key.get("/api/health").status_code == 200
    assert api_client_with_key.get("/api/imports").status_code == 200
    assert api_client_with_key.get("/api/models").status_code == 200


# ── Saved predictions (POST/GET /api/predictions) ───────────────────────────

def _seed_trained_baseline(sample_games_df):
    """Import the sample games + train a baseline into the api_client's DB so
    /api/predictions can run a real prediction."""
    import os
    from database import NBADatabase
    from predictive_models import PredictiveModels

    db = NBADatabase(db_path=os.environ["NBA_DB_PATH"])
    iid = db.register_import("games.csv", "/tmp/games.csv", record_count=len(sample_games_df))
    df = sample_games_df.copy()
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, df)
    pm = PredictiveModels(db=db)
    model_id, _ = pm.train_baseline(iid, strategy="most_frequent")
    return iid, model_id


def test_create_prediction_requires_handle(api_client):
    r = api_client.post("/api/predictions", json={
        "user_handle": "   ", "home_team": "1610612737",
        "away_team": "1610612738", "season": "2019-20", "model_type": "baseline",
    })
    assert r.status_code == 400
    assert "handle" in r.json()["detail"].lower()


def test_create_prediction_requires_model(api_client):
    r = api_client.post("/api/predictions", json={
        "user_handle": "akram", "home_team": "1610612737",
        "away_team": "1610612738", "season": "2019-20",
    })
    assert r.status_code == 400
    assert "model_id" in r.json()["detail"]


def test_list_predictions_empty(api_client):
    r = api_client.get("/api/predictions?user=nobody")
    assert r.status_code == 200
    assert r.json() == []


def test_save_and_list_prediction_round_trip(api_client, sample_games_df):
    _, model_id = _seed_trained_baseline(sample_games_df)
    teams = sorted(set(sample_games_df["home_team"]) | set(sample_games_df["away_team"]))

    r = api_client.post("/api/predictions", json={
        "user_handle": "akram", "home_team": teams[0], "away_team": teams[1],
        "season": "2019-20", "model_id": model_id,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_handle"] == "akram"
    assert body["prediction"] in ("Home Win", "Away Win")
    assert body["predicted_label"] in ("home_win", "away_win")
    assert body["model_id"] == model_id

    lst = api_client.get("/api/predictions?user=akram")
    assert lst.status_code == 200
    items = lst.json()
    assert len(items) == 1
    assert items[0]["home_team"] == teams[0]
    assert items[0]["model_name"] is not None
    assert int(items[0]["resolved"]) == 0


def test_save_prediction_requires_api_key_when_enforced(api_client_with_key):
    r = api_client_with_key.post("/api/predictions", json={
        "user_handle": "akram", "home_team": "1610612737",
        "away_team": "1610612738", "season": "2019-20", "model_type": "baseline",
    })
    assert r.status_code == 401


# ── Leaderboard (GET /api/leaderboard) ──────────────────────────────────────

def test_leaderboard_empty_on_fresh_db(api_client):
    r = api_client.get("/api/leaderboard")
    assert r.status_code == 200
    assert r.json() == []


def test_leaderboard_ranks_trained_models(api_client, sample_games_df):
    _seed_trained_baseline(sample_games_df)
    r = api_client.get("/api/leaderboard?sort=accuracy")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 1
    top = rows[0]
    assert top["rank"] == 1
    assert top["model_type"] is not None
    assert top["accuracy"] is None or 0.0 <= top["accuracy"] <= 1.0
    assert top["sorted_by"] == "accuracy"
    assert top["evaluated_games"] >= 0
    # ranks are 1..N, strictly increasing
    assert [x["rank"] for x in rows] == list(range(1, len(rows) + 1))


def test_leaderboard_sort_param_selects_metric(api_client, sample_games_df):
    _seed_trained_baseline(sample_games_df)
    assert api_client.get("/api/leaderboard?sort=cv").json()[0]["sorted_by"] == "cv_accuracy_mean"
    assert api_client.get("/api/leaderboard?sort=f1").json()[0]["sorted_by"] == "f1_score"
    # Unknown sort falls back to accuracy rather than erroring.
    assert api_client.get("/api/leaderboard?sort=bogus").json()[0]["sorted_by"] == "accuracy"


def test_leaderboard_reorders_when_metric_disagrees(api_client):
    """Two models where accuracy and CV-accuracy disagree must swap rank
    depending on the sort metric — proves the board actually re-ranks, not
    just relabels `sorted_by`."""
    import os
    from database import NBADatabase

    db = NBADatabase(db_path=os.environ["NBA_DB_PATH"])
    iid = db.register_import("x.csv", "/tmp/x.csv", record_count=0)
    # Model A: high test acc, low CV.  Model B: low test acc, high CV.
    a = db.register_model("Model A", "RandomForest", {}, model_path="/tmp/a.pkl")
    b = db.register_model("Model B", "DecisionTree", {}, model_path="/tmp/b.pkl")
    db.save_prediction_results(iid, a, accuracy=0.70, precision=0.7, recall=0.7,
                               f1=0.7, predictions=[1], actual_results=[1],
                               cv_accuracy_mean=0.50, cv_accuracy_std=0.01)
    db.save_prediction_results(iid, b, accuracy=0.60, precision=0.6, recall=0.6,
                               f1=0.6, predictions=[1], actual_results=[1],
                               cv_accuracy_mean=0.65, cv_accuracy_std=0.01)

    by_acc = api_client.get("/api/leaderboard?sort=accuracy").json()
    assert by_acc[0]["model_name"] == "Model A"  # 0.70 acc wins
    assert by_acc[1]["model_name"] == "Model B"

    by_cv = api_client.get("/api/leaderboard?sort=cv").json()
    assert by_cv[0]["model_name"] == "Model B"  # 0.65 CV wins → order flipped
    assert by_cv[1]["model_name"] == "Model A"


def test_leaderboard_scopes_by_import(api_client):
    """import_id must restrict the board to that import's models — the step-4
    gate so cross-import (different test split) accuracies aren't mixed."""
    import os
    from database import NBADatabase
    db = NBADatabase(db_path=os.environ["NBA_DB_PATH"])
    i1 = db.register_import("a.csv", "/tmp/a", 0)
    i2 = db.register_import("b.csv", "/tmp/b", 0)
    m1 = db.register_model("M1", "RandomForest", {}, model_path="/tmp/m1.pkl")
    m2 = db.register_model("M2", "DecisionTree", {}, model_path="/tmp/m2.pkl")
    db.save_prediction_results(i1, m1, accuracy=0.6, precision=0.6, recall=0.6,
                               f1=0.6, predictions=[1], actual_results=[1])
    db.save_prediction_results(i2, m2, accuracy=0.7, precision=0.7, recall=0.7,
                               f1=0.7, predictions=[1], actual_results=[1])

    assert len(api_client.get("/api/leaderboard").json()) == 2
    scoped = api_client.get(f"/api/leaderboard?import_id={i1}").json()
    assert len(scoped) == 1
    assert scoped[0]["model_name"] == "M1"


# ── Freshness + user leaderboard ────────────────────────────────────────────

def test_freshness_status_reports_unconfigured(api_client):
    # No BALLDONTLIE_API_KEY in the test env → configured is False.
    r = api_client.get("/api/freshness/status")
    assert r.status_code == 200
    assert r.json()["configured"] is False


def test_freshness_sync_rejects_unbounded_request(api_client):
    # No max_games → rejected before the configured check, pointing at the CLI.
    r = api_client.post("/api/freshness/sync", json={"seasons": [2024]})
    assert r.status_code == 400
    assert "max_games" in r.json()["detail"]


def test_freshness_sync_rejects_too_many_games(api_client):
    r = api_client.post("/api/freshness/sync", json={"seasons": [2024], "max_games": 100000})
    assert r.status_code == 400
    assert "max_games" in r.json()["detail"]


def test_freshness_sync_rejected_when_unconfigured(api_client):
    # Bounded request reaches the configured check (no key in the test env).
    r = api_client.post("/api/freshness/sync", json={"seasons": [2024], "max_games": 50})
    assert r.status_code == 400
    assert "BALLDONTLIE_API_KEY" in r.json()["detail"]


def test_user_leaderboard_empty_on_fresh_db(api_client):
    r = api_client.get("/api/user-leaderboard")
    assert r.status_code == 200
    assert r.json() == []


# ── Delete import (DELETE /api/imports/{id}) ────────────────────────────────

def test_delete_import_unknown_returns_404(api_client):
    r = api_client.delete("/api/imports/9999")
    assert r.status_code == 404


def test_delete_import_removes_it(api_client, sample_games_df):
    import os
    from database import NBADatabase
    db = NBADatabase(db_path=os.environ["NBA_DB_PATH"])
    iid = db.register_import("games.csv", "/tmp/games.csv", len(sample_games_df))
    df = sample_games_df.copy()
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, df)

    assert any(i["id"] == iid for i in api_client.get("/api/imports").json())
    r = api_client.delete(f"/api/imports/{iid}")
    assert r.status_code == 200
    assert r.json()["games_deleted"] == len(sample_games_df)
    assert not any(i["id"] == iid for i in api_client.get("/api/imports").json())


def test_delete_import_requires_api_key_when_enforced(api_client_with_key):
    r = api_client_with_key.delete("/api/imports/1")
    assert r.status_code == 401

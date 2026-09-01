"""
Auth coverage for api/main.py's verify_api_key dependency: every /api/*
route except /api/health must require a valid X-API-Key header, and the
whole thing must fail closed (reject everything) when API_KEY itself
isn't configured on the server — a missing key is not the same as "auth
disabled."
"""


class TestMissingOrWrongKey:
    def test_entities_rejects_no_key(self, client):
        r = client.get("/api/entities")
        assert r.status_code == 401

    def test_entities_rejects_wrong_key(self, client, auth_headers):
        bad_headers = {"X-API-Key": auth_headers["X-API-Key"] + "-wrong"}
        r = client.get("/api/entities", headers=bad_headers)
        assert r.status_code == 401

    def test_score_rejects_no_key(self, client):
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0})
        assert r.status_code == 401

    def test_reset_rejects_no_key(self, client):
        r = client.post("/api/entities/reset", json={"entity_id": "entity-a"})
        assert r.status_code == 401

    def test_escalation_rejects_no_key(self, client):
        r = client.get("/api/entities/entity-a/escalation")
        assert r.status_code == 401

    def test_transactions_rejects_no_key(self, client):
        r = client.get("/api/entities/entity-a/transactions")
        assert r.status_code == 401

    def test_explanations_rejects_no_key(self, client):
        r = client.get("/api/explanations/some-id")
        assert r.status_code == 401

    def test_cost_analysis_rejects_no_key(self, client):
        r = client.get("/api/cost-analysis")
        assert r.status_code == 401


class TestCorrectKey:
    def test_entities_accepts_correct_key(self, client, auth_headers):
        r = client.get("/api/entities", headers=auth_headers)
        assert r.status_code == 200

    def test_score_accepts_correct_key(self, client, auth_headers):
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        assert r.status_code == 200

    def test_reset_accepts_correct_key(self, client, auth_headers):
        r = client.post("/api/entities/reset", json={"entity_id": "entity-a"}, headers=auth_headers)
        assert r.status_code == 200


class TestHealthIsUnprotected:
    def test_health_requires_no_key(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestFailsClosedWhenUnconfigured:
    def test_unset_api_key_rejects_every_request_including_with_a_header(self, monkeypatch, sample_df):
        # Deliberately does NOT use the `client` fixture (which always sets
        # API_KEY) — this is the one scenario that must reject regardless
        # of what the caller sends, including a plausible-looking key.
        monkeypatch.delenv("API_KEY", raising=False)

        import api.main as main

        main.get_sample_data.cache_clear()
        monkeypatch.setattr(main, "get_sample_data", lambda: sample_df)

        from fastapi.testclient import TestClient
        c = TestClient(main.app)

        r = c.get("/api/entities", headers={"X-API-Key": "anything-at-all"})
        assert r.status_code == 401
        assert "not configured" in r.json()["detail"].lower()

        # Health must still work — it has no auth dependency at all.
        assert c.get("/api/health").status_code == 200

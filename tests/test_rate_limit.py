"""
POST /api/score is limited to 30/minute per caller identity (see
_rate_limit_key in api/main.py — keyed by X-API-Key when present, else
source IP). Firing 31 rapid requests in a loop exercises the real limit
window in real time — no need to actually wait a minute, since all 31
calls land well inside the same window.
"""
from unittest.mock import MagicMock

from api.main import _rate_limit_key


class TestRateLimitKey:
    def test_uses_the_api_key_when_present(self):
        request = MagicMock()
        request.headers = {"X-API-Key": "abc123"}
        assert _rate_limit_key(request) == "apikey:abc123"

    def test_different_api_keys_get_different_buckets(self):
        req1, req2 = MagicMock(), MagicMock()
        req1.headers = {"X-API-Key": "key-one"}
        req2.headers = {"X-API-Key": "key-two"}
        assert _rate_limit_key(req1) != _rate_limit_key(req2)

    def test_falls_back_to_source_ip_when_no_api_key_header(self):
        request = MagicMock()
        request.headers = {}
        request.client.host = "203.0.113.5"
        assert "203.0.113.5" in _rate_limit_key(request)


def _score(client, auth_headers):
    return client.post(
        "/api/score",
        json={"entity_id": "entity-a", "txn_index": 0},
        headers=auth_headers,
    )


class TestScoreRateLimit:
    def test_requests_within_the_limit_all_succeed(self, client, auth_headers):
        for _ in range(30):
            r = _score(client, auth_headers)
            assert r.status_code == 200

    def test_the_31st_request_within_a_minute_is_rate_limited(self, client, auth_headers):
        for _ in range(30):
            assert _score(client, auth_headers).status_code == 200

        r = _score(client, auth_headers)
        assert r.status_code == 429
        assert "rate limit" in r.json()["error"].lower()

    def test_other_routes_are_not_rate_limited(self, client, auth_headers):
        # The 30/minute limit is specific to /api/score (the expensive,
        # ML-scoring endpoint) — cheap read routes must not share its budget.
        for _ in range(35):
            r = client.get("/api/entities", headers=auth_headers)
            assert r.status_code == 200

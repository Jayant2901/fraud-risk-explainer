import json
import logging

from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from logging_utils import JsonFormatter, RequestIdFilter, RequestIDMiddleware, request_id_var


def _make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger", level=logging.WARNING, pathname="x.py", lineno=1,
        msg="something happened", args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestJsonFormatter:
    def test_output_is_valid_json_with_expected_fields(self):
        record = _make_record()
        payload = json.loads(JsonFormatter().format(record))
        assert payload["level"] == "WARNING"
        assert payload["logger"] == "test.logger"
        assert payload["message"] == "something happened"
        assert "timestamp" in payload

    def test_extra_fields_are_included_verbatim(self):
        record = _make_record(verdict_id="abc-123")
        payload = json.loads(JsonFormatter().format(record))
        assert payload["verdict_id"] == "abc-123"

    def test_request_id_defaults_to_none_when_unset(self):
        record = _make_record()  # no request_id attribute set at all
        payload = json.loads(JsonFormatter().format(record))
        assert payload["request_id"] is None

    def test_exception_info_is_included_when_present(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = _make_record()
            record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exc_info"]


class TestRequestIdFilter:
    def test_injects_the_current_contextvar_value(self):
        token = request_id_var.set("req-42")
        try:
            record = _make_record()
            RequestIdFilter().filter(record)
            assert record.request_id == "req-42"
        finally:
            request_id_var.reset(token)

    def test_defaults_to_none_outside_any_request(self):
        record = _make_record()
        RequestIdFilter().filter(record)
        assert record.request_id is None


class TestRequestIDMiddleware:
    def _app(self):
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)
        captured = {}

        def bg_task():
            captured["background_request_id"] = request_id_var.get()

        @app.get("/test")
        def route(background_tasks: BackgroundTasks):
            captured["route_request_id"] = request_id_var.get()
            background_tasks.add_task(bg_task)
            return {"ok": True}

        return app, captured

    def test_generates_a_request_id_and_returns_it_as_a_header(self):
        app, _ = self._app()
        r = TestClient(app).get("/test")
        assert r.headers["x-request-id"]

    def test_forwards_a_caller_supplied_request_id(self):
        app, _ = self._app()
        r = TestClient(app).get("/test", headers={"X-Request-ID": "caller-provided-id"})
        assert r.headers["x-request-id"] == "caller-provided-id"

    def test_different_requests_get_different_ids(self):
        app, _ = self._app()
        client = TestClient(app)
        r1 = client.get("/test")
        r2 = client.get("/test")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    def test_contextvar_is_set_during_the_route_and_its_background_task(self):
        app, captured = self._app()
        r = TestClient(app).get("/test")
        request_id = r.headers["x-request-id"]
        assert captured["route_request_id"] == request_id
        assert captured["background_request_id"] == request_id

    def test_contextvar_is_unset_again_after_the_request_completes(self):
        app, _ = self._app()
        TestClient(app).get("/test")
        assert request_id_var.get() is None


class TestMetricsEndpoint:
    def test_metrics_is_reachable_without_an_api_key(self, client):
        # /metrics is intentionally NOT behind verify_api_key (see
        # api/main.py) — Prometheus scraping conventions expect
        # network-level access control, not an application API key.
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_body_looks_like_prometheus_text_format(self, client):
        r = client.get("/metrics")
        assert "# HELP" in r.text
        assert "# TYPE" in r.text

    def test_domain_metrics_appear_alongside_the_http_ones(self, client):
        r = client.get("/metrics")
        assert "riskmgr_review_queue_pending" in r.text
        assert "riskmgr_llm_breaker_open" in r.text

    def test_a_scored_transaction_is_reflected_in_the_decisions_counter(self, client, auth_headers):
        client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)

        body = client.get("/metrics").text

        # FakeExplainer (conftest.py) returns risk_score=42.0, which
        # decide_action() turns into REVIEW for an unescalated entity.
        assert 'riskmgr_decisions_total{action="REVIEW"}' in body

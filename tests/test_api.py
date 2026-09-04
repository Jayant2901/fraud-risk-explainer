"""
Route coverage for api/main.py using FastAPI's TestClient against the
fully-faked app from conftest.py (fake sample data, fake explainer, fake
LLM agent) — no trained model, no dataset, no GEMINI_API_KEY needed.
"""
# Imported from the module that owns it rather than via api.main: the
# scoring logic now lives in src/scoring_service.py, so api/main.py no
# longer imports decide_action at all.
from decision_rules import decide_action


class TestDecideAction:
    """
    Direct unit tests for the one function that actually gates a
    transaction (see the "Do not touch" note in the module docstring) —
    cheap to test as a pure function, and the most important piece of
    business logic in the file, so it shouldn't only be covered
    incidentally through whatever risk_score the /api/score tests
    happen to use.
    """

    def test_high_score_blocks_regardless_of_escalation(self):
        assert decide_action(85.0, {"state": "NORMAL"})["action"] == "BLOCK"
        assert decide_action(85.0, {"state": "ELEVATED"})["action"] == "BLOCK"

    def test_high_score_block_is_never_reported_as_escalated(self):
        # BLOCK at >=80 is what the score alone already demands — it's not
        # an escalation due to history, even if the entity happens to be
        # ELEVATED too.
        assert decide_action(85.0, {"state": "ELEVATED"})["escalated_due_to_history"] is False

    def test_mid_score_reviews_when_not_elevated(self):
        result = decide_action(55.0, {"state": "NORMAL"})
        assert result["action"] == "REVIEW"
        assert result["escalated_due_to_history"] is False

    def test_mid_score_escalates_to_block_when_elevated(self):
        result = decide_action(55.0, {"state": "ELEVATED"})
        assert result["action"] == "BLOCK"
        assert result["escalated_due_to_history"] is True

    def test_mid_score_does_not_escalate_on_watch_alone(self):
        # Only ELEVATED escalates per the LLM system prompt's own stated
        # rules (src/llm_agent.py) — WATCH influences nothing here.
        result = decide_action(55.0, {"state": "WATCH"})
        assert result["action"] == "REVIEW"
        assert result["escalated_due_to_history"] is False

    def test_low_score_allows_when_not_elevated(self):
        result = decide_action(10.0, {"state": "NORMAL"})
        assert result["action"] == "ALLOW"
        assert result["escalated_due_to_history"] is False

    def test_low_score_escalates_to_review_when_elevated(self):
        result = decide_action(10.0, {"state": "ELEVATED"})
        assert result["action"] == "REVIEW"
        assert result["escalated_due_to_history"] is True

    def test_boundary_scores_are_inclusive_on_the_higher_band(self):
        assert decide_action(80.0, {"state": "NORMAL"})["action"] == "BLOCK"
        assert decide_action(40.0, {"state": "NORMAL"})["action"] == "REVIEW"
        assert decide_action(39.9, {"state": "NORMAL"})["action"] == "ALLOW"


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_reports_the_llm_circuit_breaker_state(self, client):
        body = client.get("/api/health").json()

        assert body["llm"]["state"] in {"closed", "open", "unknown"}
        assert "consecutive_failures" in body["llm"]
        assert "seconds_until_retry" in body["llm"]

    def test_reflects_an_open_breaker(self, client, monkeypatch):
        import api.main as main
        from circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("llm-health-test", failure_threshold=1, cooldown_seconds=60)
        breaker.record_failure()

        class AgentWithOpenBreaker:
            @property
            def breaker(self):
                return breaker

        monkeypatch.setattr(main, "get_agent", lambda: AgentWithOpenBreaker())

        body = client.get("/api/health").json()

        assert body["llm"]["state"] == "open"
        assert body["llm"]["consecutive_failures"] == 1
        assert body["llm"]["seconds_until_retry"] > 0
        # Explanations degrading is deliberately not this service failing:
        # scoring is unaffected, so overall status stays ok.
        assert body["status"] == "ok"

    def test_reports_feature_store_coverage(self, client):
        """Seeded vs. live counts: a store that looks populated but was
        never seeded would score early transactions against empty history,
        and nothing else would surface that."""
        body = client.get("/api/health").json()

        assert "entities_tracked" in body["feature_store"]
        assert "fingerprints_tracked" in body["feature_store"]
        assert "seeded_rows" in body["feature_store"]

    def test_stays_up_when_the_agent_cannot_be_constructed(self, client, monkeypatch):
        import api.main as main

        def exploding_agent():
            raise RuntimeError("no credentials")

        monkeypatch.setattr(main, "get_agent", exploding_agent)

        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["llm"]["state"] == "unknown"

    def test_reports_whether_an_escalation_webhook_is_configured(self, client, monkeypatch):
        monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)

        body = client.get("/api/health").json()

        assert body["escalation_alerts"]["webhook_configured"] is False


class TestListEntities:
    def test_returns_unique_entity_ids(self, client, auth_headers):
        r = client.get("/api/entities", headers=auth_headers)
        assert r.status_code == 200
        assert set(r.json()["entities"]) == {"entity-a", "entity-b"}


class TestListTransactions:
    def test_happy_path_returns_transactions_in_order(self, client, auth_headers):
        r = client.get("/api/entities/entity-a/transactions", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["entity_id"] == "entity-a"
        assert body["count"] == 2
        assert [t["index"] for t in body["transactions"]] == [0, 1]
        assert body["transactions"][0]["TransactionAmt"] == 100.0

    def test_unknown_entity_returns_404(self, client, auth_headers):
        r = client.get("/api/entities/does-not-exist/transactions", headers=auth_headers)
        assert r.status_code == 404


class TestEscalation:
    def test_unknown_entity_defaults_to_normal(self, client, auth_headers):
        r = client.get("/api/entities/entity-a/escalation", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["state"] == "NORMAL"

    def test_reflects_recorded_verdicts_after_scoring(self, client, auth_headers):
        for txn_index in (0, 1):
            client.post(
                "/api/score",
                json={"entity_id": "entity-a", "txn_index": txn_index},
                headers=auth_headers,
            )
        r = client.get("/api/entities/entity-a/escalation", headers=auth_headers)
        assert r.json()["recent_verdict_count"] == 2


class TestResetEntity:
    def test_reset_clears_recorded_verdicts(self, client, auth_headers):
        client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        assert client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()["recent_verdict_count"] == 1

        r = client.post("/api/entities/reset", json={"entity_id": "entity-a"}, headers=auth_headers)
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        assert client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()["recent_verdict_count"] == 0


class TestScore:
    def test_happy_path_returns_decision_and_verdict_id(self, client, auth_headers):
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["risk_score"] == 42.0
        assert body["decision"]["action"] in {"ALLOW", "REVIEW", "BLOCK"}
        assert "verdict_id" in body
        assert "escalation_before" in body

    def test_records_a_verdict_synchronously(self, client, auth_headers):
        # The decision must already be reflected in escalation state before
        # the score response even returns — it's not waiting on the
        # background LLM explanation task.
        client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        state = client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()
        assert state["recent_verdict_count"] == 1

    def test_unknown_entity_returns_404(self, client, auth_headers):
        r = client.post("/api/score", json={"entity_id": "nope", "txn_index": 0}, headers=auth_headers)
        assert r.status_code == 404

    def test_out_of_range_txn_index_returns_400(self, client, auth_headers):
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 99}, headers=auth_headers)
        assert r.status_code == 400

    def test_negative_txn_index_returns_400(self, client, auth_headers):
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": -1}, headers=auth_headers)
        assert r.status_code == 400

    def test_schedules_a_background_explanation(self, client, auth_headers):
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        verdict_id = r.json()["verdict_id"]
        # TestClient runs background tasks synchronously before returning,
        # so this is already resolved by the time we get here.
        exp = client.get(f"/api/explanations/{verdict_id}", headers=auth_headers)
        assert exp.status_code == 200
        assert exp.json()["status"] == "ready"
        assert exp.json()["verdict"]["action"] == "REVIEW"  # from FakeAgent

    def test_explanation_never_hangs_at_pending_if_the_agent_construction_crashes(self, client, auth_headers, monkeypatch):
        """
        Regression test for a real production bug: RiskExplainerAgent used
        to construct its API client eagerly, outside any try/except — a
        missing/invalid credential crashed the background task with an
        unhandled exception, and since nothing ever wrote a terminal
        status for that verdict_id, the frontend polled "pending" forever.
        _generate_explanation's own try/except (api/main.py) is the
        backstop for exactly this: whatever raises inside get_agent()/
        agent.explain(), the explanation must still resolve to "ready"
        with a safe fallback, never hang.
        """
        import api.main as main

        def explode():
            raise RuntimeError("simulated crash constructing the LLM client")

        monkeypatch.setattr(main, "get_agent", explode)

        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        verdict_id = r.json()["verdict_id"]

        exp = client.get(f"/api/explanations/{verdict_id}", headers=auth_headers)
        assert exp.status_code == 200
        body = exp.json()
        assert body["status"] == "ready"  # not stuck at "pending"
        assert body["verdict"]["action"] == "REVIEW"


class TestScoreIdempotency:
    """
    The project's own stated differentiator (README point 5): a retried
    POST /api/score with the same Idempotency-Key must not score/record
    the transaction twice. Verified two ways — identical response body,
    and (behaviorally, not via a mock) that the entity's escalation
    history only grew by one verdict, not two.
    """

    def test_same_idempotency_key_returns_the_identical_cached_response(self, client, auth_headers):
        headers = {**auth_headers, "Idempotency-Key": "retry-key-1"}
        payload = {"entity_id": "entity-a", "txn_index": 0}

        r1 = client.post("/api/score", json=payload, headers=headers)
        r2 = client.post("/api/score", json=payload, headers=headers)

        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json() == r2.json()
        assert r1.json()["verdict_id"] == r2.json()["verdict_id"]

    def test_same_idempotency_key_records_the_verdict_only_once(self, client, auth_headers):
        headers = {**auth_headers, "Idempotency-Key": "retry-key-2"}
        payload = {"entity_id": "entity-a", "txn_index": 0}

        client.post("/api/score", json=payload, headers=headers)
        client.post("/api/score", json=payload, headers=headers)  # retry with same key

        state = client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()
        assert state["recent_verdict_count"] == 1  # not 2

    def test_different_idempotency_keys_are_scored_and_recorded_independently(self, client, auth_headers):
        payload = {"entity_id": "entity-a", "txn_index": 0}
        client.post("/api/score", json=payload, headers={**auth_headers, "Idempotency-Key": "key-a"})
        client.post("/api/score", json=payload, headers={**auth_headers, "Idempotency-Key": "key-b"})

        state = client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()
        assert state["recent_verdict_count"] == 2

    def test_omitting_idempotency_key_does_not_dedupe(self, client, auth_headers):
        payload = {"entity_id": "entity-a", "txn_index": 0}
        r1 = client.post("/api/score", json=payload, headers=auth_headers)
        r2 = client.post("/api/score", json=payload, headers=auth_headers)

        assert r1.json()["verdict_id"] != r2.json()["verdict_id"]
        state = client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()
        assert state["recent_verdict_count"] == 2


class TestExplanations:
    def test_unknown_verdict_id_returns_404(self, client, auth_headers):
        r = client.get("/api/explanations/does-not-exist", headers=auth_headers)
        assert r.status_code == 404

    def test_ready_explanation_carries_the_agents_verdict(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.get(f"/api/explanations/{verdict_id}", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["verdict"]["explanation"] == "fake explanation"


class TestReviewQueue:
    """FakeExplainer (conftest.py) returns a fixed risk_score=42.0, which
    decide_action() turns into REVIEW (non-ALLOW) for an unescalated
    entity — so every /api/score call in these tests lands in the queue
    without needing to juggle escalation state."""

    def test_flagged_verdict_appears_in_the_pending_queue(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]
        assert score_resp.json()["decision"]["action"] == "REVIEW"

        r = client.get("/api/review-queue?status=pending", headers=auth_headers)
        assert r.status_code == 200
        items = r.json()["items"]
        assert any(i["verdict_id"] == verdict_id for i in items)

    def test_allowed_verdicts_never_reach_the_queue(self, client, auth_headers, monkeypatch):
        import api.main as main
        from tests.conftest import FakeExplainer

        monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer(risk_score=5.0))

        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        assert score_resp.json()["decision"]["action"] == "ALLOW"

        items = client.get("/api/review-queue?status=pending", headers=auth_headers).json()["items"]
        assert items == []

    def test_pending_items_are_sorted_by_risk_score_descending(self, client, auth_headers, monkeypatch):
        import api.main as main
        from tests.conftest import FakeExplainer

        monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer(risk_score=45.0))
        client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)

        monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer(risk_score=90.0))
        client.post("/api/score", json={"entity_id": "entity-b", "txn_index": 0}, headers=auth_headers)

        items = client.get("/api/review-queue?status=pending", headers=auth_headers).json()["items"]
        assert [i["risk_score"] for i in items] == [90.0, 45.0]

    def test_disposing_a_flagged_verdict_removes_it_from_pending(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.post(
            f"/api/review-queue/{verdict_id}/disposition",
            json={"disposition": "CONFIRMED_FRAUD"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["disposition"] == "CONFIRMED_FRAUD"

        items = client.get("/api/review-queue?status=pending", headers=auth_headers).json()["items"]
        assert items == []

    def test_disposing_twice_returns_409_and_keeps_the_first_disposition(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        client.post(
            f"/api/review-queue/{verdict_id}/disposition",
            json={"disposition": "CONFIRMED_FRAUD"},
            headers=auth_headers,
        )
        r = client.post(
            f"/api/review-queue/{verdict_id}/disposition",
            json={"disposition": "FALSE_POSITIVE"},
            headers=auth_headers,
        )
        assert r.status_code == 409

    def test_disposing_an_unknown_verdict_id_returns_404(self, client, auth_headers):
        r = client.post(
            "/api/review-queue/does-not-exist/disposition",
            json={"disposition": "CONFIRMED_FRAUD"},
            headers=auth_headers,
        )
        assert r.status_code == 404

    def test_invalid_disposition_value_is_rejected(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.post(
            f"/api/review-queue/{verdict_id}/disposition",
            json={"disposition": "MAYBE"},
            headers=auth_headers,
        )
        assert r.status_code == 422

    def test_metrics_reflect_actual_dispositions(self, client, auth_headers):
        r1 = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        r2 = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 1}, headers=auth_headers)

        client.post(
            f"/api/review-queue/{r1.json()['verdict_id']}/disposition",
            json={"disposition": "CONFIRMED_FRAUD"},
            headers=auth_headers,
        )
        client.post(
            f"/api/review-queue/{r2.json()['verdict_id']}/disposition",
            json={"disposition": "FALSE_POSITIVE"},
            headers=auth_headers,
        )

        metrics = client.get("/api/review-queue/metrics", headers=auth_headers).json()
        assert metrics["total_disposed"] == 2
        assert metrics["overall_precision"] == 0.5

    def test_unsupported_status_filter_returns_400(self, client, auth_headers):
        r = client.get("/api/review-queue?status=disposed", headers=auth_headers)
        assert r.status_code == 400

    def test_new_items_carry_a_created_at_timestamp_and_empty_notes(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        items = client.get("/api/review-queue?status=pending", headers=auth_headers).json()["items"]
        item = next(i for i in items if i["verdict_id"] == verdict_id)
        assert item["created_at"] is not None
        assert item["notes"] == []


class TestReviewQueueNotes:
    def test_adding_a_note_returns_it_and_it_appears_on_the_item(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.post(
            f"/api/review-queue/{verdict_id}/notes",
            json={"author": "Alice", "text": "Looks like a stolen card."},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["author"] == "Alice"
        assert r.json()["text"] == "Looks like a stolen card."

        items = client.get("/api/review-queue?status=pending", headers=auth_headers).json()["items"]
        item = next(i for i in items if i["verdict_id"] == verdict_id)
        assert item["notes"] == [r.json()]

    def test_author_defaults_to_reviewer_when_omitted(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.post(
            f"/api/review-queue/{verdict_id}/notes", json={"text": "note text"}, headers=auth_headers
        )
        assert r.status_code == 200
        assert r.json()["author"] == "Reviewer"

    def test_unknown_verdict_id_returns_404(self, client, auth_headers):
        r = client.post(
            "/api/review-queue/does-not-exist/notes",
            json={"text": "note text"},
            headers=auth_headers,
        )
        assert r.status_code == 404

    def test_empty_text_is_rejected(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.post(
            f"/api/review-queue/{verdict_id}/notes", json={"text": ""}, headers=auth_headers
        )
        assert r.status_code == 422

    def test_overlong_text_is_rejected(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.post(
            f"/api/review-queue/{verdict_id}/notes",
            json={"text": "x" * 2001},
            headers=auth_headers,
        )
        assert r.status_code == 422


class TestReviewQueueRelated:
    def test_unknown_verdict_id_returns_404(self, client, auth_headers):
        r = client.get("/api/review-queue/does-not-exist/related", headers=auth_headers)
        assert r.status_code == 404

    def test_no_related_items_returns_empty_list(self, client, auth_headers):
        score_resp = client.post(
            "/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers
        )
        verdict_id = score_resp.json()["verdict_id"]

        r = client.get(f"/api/review-queue/{verdict_id}/related", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["items"] == []

    def test_only_same_entity_items_are_related(self, client, auth_headers):
        r1 = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        r2 = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 1}, headers=auth_headers)
        r3 = client.post("/api/score", json={"entity_id": "entity-b", "txn_index": 0}, headers=auth_headers)

        related = client.get(
            f"/api/review-queue/{r1.json()['verdict_id']}/related", headers=auth_headers
        ).json()["items"]
        related_ids = {i["verdict_id"] for i in related}
        assert r2.json()["verdict_id"] in related_ids
        assert r3.json()["verdict_id"] not in related_ids
        assert r1.json()["verdict_id"] not in related_ids


class TestScoreCustom:
    """POST /api/score-custom — scoring a transaction that isn't one of
    the ~30 cached historical entities. FakeExplainer (conftest.py)
    returns a fixed risk_score=42.0 regardless of the txn dict passed
    in, which decide_action() turns into REVIEW for an unescalated
    entity — so most of these land in the review queue too, same as
    /api/score."""

    def test_minimal_payload_returns_a_valid_score_and_decision(self, client, auth_headers):
        r = client.post("/api/score-custom", json={"TransactionAmt": 250.0}, headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["risk_score"] == 42.0
        assert body["decision"]["action"] in {"ALLOW", "REVIEW", "BLOCK"}
        assert "verdict_id" in body

    def test_all_optional_fields_omitted_does_not_crash(self, client, auth_headers):
        r = client.post("/api/score-custom", json={"TransactionAmt": 10.0}, headers=auth_headers)
        assert r.status_code == 200

    def test_full_payload_is_accepted(self, client, auth_headers):
        r = client.post(
            "/api/score-custom",
            json={
                "TransactionAmt": 500.0,
                "ProductCD": "W",
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "",
                "DeviceType": "mobile",
                "addr1": 300,
                "addr2": 87,
                "hour_of_day": 14,
            },
            headers=auth_headers,
        )
        assert r.status_code == 200

    def test_missing_required_field_returns_422(self, client, auth_headers):
        r = client.post("/api/score-custom", json={"ProductCD": "W"}, headers=auth_headers)
        assert r.status_code == 422

    def test_hour_of_day_out_of_range_returns_422(self, client, auth_headers):
        r = client.post(
            "/api/score-custom", json={"TransactionAmt": 100.0, "hour_of_day": 24}, headers=auth_headers
        )
        assert r.status_code == 422

    def test_without_attach_escalation_is_normal_baseline(self, client, auth_headers):
        r = client.post("/api/score-custom", json={"TransactionAmt": 100.0}, headers=auth_headers)
        assert r.json()["escalation_before"]["state"] == "NORMAL"
        assert r.json()["escalation_before"]["recent_verdict_count"] == 0

    def test_with_attach_escalation_matches_that_entitys_real_current_state(self, client, auth_headers):
        # Push entity-a into a non-empty history first via a real replay score.
        client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0}, headers=auth_headers)
        state_before = client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()

        r = client.post(
            "/api/score-custom",
            json={"TransactionAmt": 100.0, "attach_to_entity_id": "entity-a"},
            headers=auth_headers,
        )
        assert r.json()["escalation_before"] == state_before

    def test_without_attach_no_entity_history_is_ever_touched(self, client, auth_headers):
        # Score the same hypothetical transaction twice with no attach —
        # nothing should accumulate anywhere.
        payload = {"TransactionAmt": 100.0}
        client.post("/api/score-custom", json=payload, headers=auth_headers)
        client.post("/api/score-custom", json=payload, headers=auth_headers)

        for entity_id in ("entity-a", "entity-b"):
            state = client.get(f"/api/entities/{entity_id}/escalation", headers=auth_headers).json()
            assert state["recent_verdict_count"] == 0

    def test_with_attach_records_the_verdict_into_that_entitys_history(self, client, auth_headers):
        client.post(
            "/api/score-custom",
            json={"TransactionAmt": 100.0, "attach_to_entity_id": "entity-a"},
            headers=auth_headers,
        )
        state = client.get("/api/entities/entity-a/escalation", headers=auth_headers).json()
        assert state["recent_verdict_count"] == 1

    def test_reuses_the_same_decide_action_pipeline_as_score(self, client, auth_headers):
        # FakeExplainer's fixed risk_score=42.0 -> decide_action() ->
        # REVIEW for an unescalated entity, identical to /api/score.
        r = client.post("/api/score-custom", json={"TransactionAmt": 100.0}, headers=auth_headers)
        assert r.json()["decision"]["action"] == "REVIEW"

    def test_flagged_custom_verdict_reaches_the_review_queue(self, client, auth_headers):
        r = client.post("/api/score-custom", json={"TransactionAmt": 100.0}, headers=auth_headers)
        verdict_id = r.json()["verdict_id"]

        items = client.get("/api/review-queue?status=pending", headers=auth_headers).json()["items"]
        assert any(i["verdict_id"] == verdict_id for i in items)

    def test_schedules_a_background_explanation(self, client, auth_headers):
        r = client.post("/api/score-custom", json={"TransactionAmt": 100.0}, headers=auth_headers)
        verdict_id = r.json()["verdict_id"]
        exp = client.get(f"/api/explanations/{verdict_id}", headers=auth_headers)
        assert exp.status_code == 200
        assert exp.json()["status"] == "ready"

    def test_requires_api_key(self, client):
        r = client.post("/api/score-custom", json={"TransactionAmt": 100.0})
        assert r.status_code == 401


class TestCostAnalysis:
    def test_returns_defaults_when_no_params_given(self, client, auth_headers):
        r = client.get("/api/cost-analysis", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "defaults" in body
        assert body["params"]["fraud_loss"] == body["defaults"]["avg_fraud_loss"]

    def test_custom_params_are_echoed_back(self, client, auth_headers):
        r = client.get("/api/cost-analysis?fraud_loss=1000&fp_cost=50", headers=auth_headers)
        assert r.json()["params"] == {"fraud_loss": 1000.0, "fp_cost": 50.0}

    def test_headline_is_null_when_no_cost_summary_exists_yet(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main

        monkeypatch.setattr(main, "COST_SUMMARY_PATH", str(tmp_path / "does-not-exist.json"))

        r = client.get("/api/cost-analysis", headers=auth_headers)
        body = r.json()
        assert body["headline_monthly_savings_estimate"] is None
        assert body["headline_basis"] is None

    def test_headline_is_computed_from_a_real_cost_summary_file(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main
        import json

        summary_path = tmp_path / "cost_summary.json"
        summary_path.write_text(json.dumps({
            "estimated_savings": 1000.0,
            "estimated_savings_pct": 8.8,
            "n_test_transactions": 1000,
        }))
        monkeypatch.setattr(main, "COST_SUMMARY_PATH", str(summary_path))

        r = client.get("/api/cost-analysis", headers=auth_headers)
        body = r.json()
        # Rs 1,000 / 1,000 txns = Rs 1/txn * 500,000 assumed monthly volume
        assert body["headline_monthly_savings_estimate"] == 500_000.0
        assert "illustrative" in body["headline_basis"].lower()

    def test_cost_curve_is_empty_when_none_has_been_generated(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main

        monkeypatch.setattr(main, "COST_CURVE_PATH", str(tmp_path / "does-not-exist.json"))

        r = client.get("/api/cost-analysis", headers=auth_headers)

        assert r.json()["cost_curve"] == []

    def test_cost_curve_total_cost_follows_the_requested_assumptions(
        self, client, auth_headers, monkeypatch, tmp_path
    ):
        import api.main as main
        import json

        curve_path = tmp_path / "cost_curve.json"
        curve_path.write_text(json.dumps([
            {"threshold": 0.1, "false_negatives": 2, "false_positives": 10},
            {"threshold": 0.9, "false_negatives": 20, "false_positives": 1},
        ]))
        monkeypatch.setattr(main, "COST_CURVE_PATH", str(curve_path))

        r = client.get("/api/cost-analysis?fraud_loss=1000&fp_cost=50", headers=auth_headers)

        # Same arithmetic as cost_analysis.cost_curve: fn*loss + fp*fp_cost.
        assert r.json()["cost_curve"] == [
            {"threshold": 0.1, "total_cost": 2 * 1000 + 10 * 50},
            {"threshold": 0.9, "total_cost": 20 * 1000 + 1 * 50},
        ]

    def test_exposes_the_live_decision_thresholds(self, client, auth_headers):
        import api.main as main

        r = client.get("/api/cost-analysis", headers=auth_headers)

        # The same object the scoring path decides with, not a copy.
        assert r.json()["decision_thresholds"] == main.get_decision_thresholds()

    def test_exposes_the_real_escalation_cutoffs(self, client, auth_headers):
        import entity_memory

        r = client.get("/api/cost-analysis", headers=auth_headers)

        assert r.json()["escalation_cutoffs"] == {
            "watch": entity_memory.DEFAULT_WATCH_PRESSURE_THRESHOLD,
            "elevated": entity_memory.DEFAULT_ELEVATED_PRESSURE_THRESHOLD,
        }

    def test_roc_auc_comes_from_the_cost_summary_file(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main
        import json

        summary_path = tmp_path / "cost_summary.json"
        summary_path.write_text(json.dumps({
            "estimated_savings": 1000.0,
            "estimated_savings_pct": 8.8,
            "n_test_transactions": 1000,
            "roc_auc": 0.9123,
        }))
        monkeypatch.setattr(main, "COST_SUMMARY_PATH", str(summary_path))

        r = client.get("/api/cost-analysis", headers=auth_headers)

        assert r.json()["roc_auc"] == 0.9123

    def test_roc_auc_is_null_for_a_summary_written_before_it_was_recorded(
        self, client, auth_headers, monkeypatch, tmp_path
    ):
        import api.main as main
        import json

        summary_path = tmp_path / "cost_summary.json"
        summary_path.write_text(json.dumps({
            "estimated_savings": 1000.0,
            "estimated_savings_pct": 8.8,
            "n_test_transactions": 1000,
        }))
        monkeypatch.setattr(main, "COST_SUMMARY_PATH", str(summary_path))

        r = client.get("/api/cost-analysis", headers=auth_headers)

        assert r.json()["roc_auc"] is None


class TestVerdictStream:
    """GET /api/verdicts/{verdict_id}/stream — SSE.

    Parsed loosely (event: / data: lines) rather than with an SSE client,
    so these assert the wire format the browser's EventSource actually
    consumes.
    """

    def _events(self, raw: str) -> list[tuple[str, dict]]:
        import json as _json

        out = []
        for block in raw.split("\n\n"):
            name = data = None
            for line in block.splitlines():
                if line.startswith("event: "):
                    name = line[len("event: "):]
                elif line.startswith("data: "):
                    data = _json.loads(line[len("data: "):])
            if name is not None:
                out.append((name, data))
        return out

    def test_unknown_verdict_id_is_404(self, client, auth_headers):
        r = client.get("/api/verdicts/nope/stream", headers=auth_headers)
        assert r.status_code == 404

    def test_requires_an_api_key(self, client):
        assert client.get("/api/verdicts/any/stream").status_code == 401

    def test_accepts_the_key_as_a_query_parameter_for_eventsource(self, client):
        """EventSource cannot set headers, so this one endpoint also takes
        the key as a query param — see verify_api_key_or_query."""
        import api.main as main
        from conftest import TEST_API_KEY

        main._explanations_cache.put("v-q", {"status": "ready", "verdict": {"action": "ALLOW"}})

        r = client.get(f"/api/verdicts/v-q/stream?api_key={TEST_API_KEY}")

        assert r.status_code == 200

    def test_rejects_a_wrong_key_in_the_query_parameter(self, client):
        assert client.get("/api/verdicts/any/stream?api_key=wrong").status_code == 401

    def _publish_after_connect(self, main, verdict_id, messages):
        """Publish once the SSE handler has subscribed.

        Pub/sub has no retention, so publishing before the subscriber
        attaches would drop the messages — which is exactly the real
        ordering too: the client connects, then the model produces text.
        """
        import threading
        import time

        def publish():
            time.sleep(0.2)
            for message in messages:
                main._explanation_bus.publish(verdict_id, message)

        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        return thread

    def test_emits_decision_before_any_explanation_event(self, client, auth_headers):
        import api.main as main

        main._explanations_cache.put("v-pending", {"status": "pending"})
        self._publish_after_connect(main, "v-pending", [
            {"type": "delta", "text": "Because "},
            {"type": "complete", "verdict": {"action": "REVIEW"}},
        ])

        r = client.get("/api/verdicts/v-pending/stream", headers=auth_headers)

        events = self._events(r.text)
        assert events[0][0] == "decision"
        assert [name for name, _ in events] == [
            "decision",
            "explanation_delta",
            "explanation_complete",
        ]

    def test_deltas_carry_the_incremental_text(self, client, auth_headers):
        import api.main as main

        main._explanations_cache.put("v-deltas", {"status": "pending"})
        self._publish_after_connect(main, "v-deltas", [
            {"type": "delta", "text": "The card "},
            {"type": "delta", "text": "has three "},
            {"type": "delta", "text": "prior blocks."},
            {"type": "complete", "verdict": {"action": "BLOCK"}},
        ])

        r = client.get("/api/verdicts/v-deltas/stream", headers=auth_headers)

        events = self._events(r.text)
        text = "".join(d["text"] for name, d in events if name == "explanation_delta")
        assert text == "The card has three prior blocks."

    def test_a_verdict_that_completes_while_connected_still_terminates(self, client, auth_headers, monkeypatch):
        """Pub/sub has no retention: if the verdict lands in the window
        between the handler's cache check and its subscription, the client
        would hang forever. The idle tick re-reads the cache to close that
        race."""
        import api.main as main
        import explanation_bus

        monkeypatch.setattr(explanation_bus, "POLL_TIMEOUT_SECONDS", 0.05)
        main._explanations_cache.put("v-raced", {"status": "pending"})
        # Written directly to the cache — no message is ever published.
        verdict = {"explanation": "raced", "action": "ALLOW",
                   "escalated_due_to_history": False, "rationale": "r"}
        main._explanations_cache.put("v-raced", {"status": "ready", "verdict": verdict})

        r = client.get("/api/verdicts/v-raced/stream", headers=auth_headers)

        events = self._events(r.text)
        assert events[-1] == ("explanation_complete", verdict)

    def test_a_stream_whose_producer_died_closes_with_a_terminal_error(self, client, auth_headers, monkeypatch):
        import api.main as main
        import explanation_bus

        monkeypatch.setattr(explanation_bus, "POLL_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(main, "STREAM_MAX_IDLE_TICKS", 2)
        main._explanations_cache.put("v-orphan", {"status": "pending"})

        r = client.get("/api/verdicts/v-orphan/stream", headers=auth_headers)

        events = self._events(r.text)
        assert events[-1][0] == "error"
        assert events[-1][1]["action"] == "REVIEW"
        assert ": keepalive" in r.text

    def test_an_already_finished_verdict_replays_its_terminal_event(self, client, auth_headers):
        """A client reconnecting after the explanation landed must not
        hang waiting for deltas that will never come again."""
        import api.main as main

        verdict = {
            "explanation": "done",
            "action": "ALLOW",
            "escalated_due_to_history": False,
            "rationale": "r",
        }
        main._explanations_cache.put("v-done", {"status": "ready", "verdict": verdict})

        r = client.get("/api/verdicts/v-done/stream", headers=auth_headers)

        events = self._events(r.text)
        assert [name for name, _ in events] == ["decision", "explanation_complete"]
        assert events[-1][1] == verdict

    def test_the_completed_verdict_matches_the_polling_endpoints_payload(self, client, auth_headers):
        """SSE and the polling fallback must deliver the same object —
        the polling endpoint stays as the transport for proxied clients."""
        import api.main as main

        verdict = {
            "explanation": "same",
            "action": "REVIEW",
            "escalated_due_to_history": True,
            "rationale": "r",
        }
        main._explanations_cache.put("v-same", {"status": "ready", "verdict": verdict})

        streamed = self._events(
            client.get("/api/verdicts/v-same/stream", headers=auth_headers).text
        )[-1][1]
        polled = client.get("/api/explanations/v-same", headers=auth_headers).json()

        assert streamed == polled["verdict"]

    def test_declares_the_event_stream_content_type(self, client, auth_headers):
        import api.main as main

        main._explanations_cache.put("v-ct", {"status": "ready", "verdict": {"action": "ALLOW"}})

        r = client.get("/api/verdicts/v-ct/stream", headers=auth_headers)

        assert r.headers["content-type"].startswith("text/event-stream")
        # Nginx would otherwise buffer the whole stream to completion.
        assert r.headers["x-accel-buffering"] == "no"


class TestTransactionEventIngestion:
    """POST /api/events/transaction — the webhook-shaped entry point.

    The endpoint must accept and return fast without scoring: scoring
    happens in src/stream_consumer.py, off the stream.
    """

    def _redis_backed(self, monkeypatch):
        """Swap the app's Redis client (normally None in tests) for a fake
        one, plus a dedup cache backed by it."""
        import api.main as main
        import fakeredis
        from redis_utils import KeyedCache

        client = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(main, "_redis_client", client)
        monkeypatch.setattr(
            main,
            "_event_dedup_cache",
            KeyedCache(client, prefix="riskmgr:events:seen", ttl_seconds=60),
        )
        return client

    def _payload(self, event_id="evt-1", **overrides):
        return {"event_id": event_id, "TransactionAmt": 100.0, "entity_id": "entity-a", **overrides}

    def test_accepts_an_event_with_202_and_a_pollable_verdict_id(self, client, auth_headers, monkeypatch):
        self._redis_backed(monkeypatch)

        r = client.post("/api/events/transaction", json=self._payload(), headers=auth_headers)

        assert r.status_code == 202
        body = r.json()
        assert body["event_id"] == "evt-1"
        assert body["verdict_id"]
        assert body["status"] == "accepted"
        assert body["duplicate"] is False

    def test_the_event_is_published_to_the_stream_not_scored_inline(self, client, auth_headers, monkeypatch):
        import event_stream

        redis_client = self._redis_backed(monkeypatch)

        r = client.post("/api/events/transaction", json=self._payload(), headers=auth_headers)

        queued = event_stream.read_events(redis_client, "test", block_ms=0)
        assert len(queued) == 1
        event = queued[0][1]
        assert event["event_id"] == "evt-1"
        assert event["entity_id"] == "entity-a"
        assert event["transaction"]["TransactionAmt"] == 100.0
        # The verdict_id handed to the caller is the one queued, so the
        # caller can poll for exactly this event's result.
        assert event["verdict_id"] == r.json()["verdict_id"]

    def test_a_replayed_event_id_produces_no_second_verdict_or_stream_entry(
        self, client, auth_headers, monkeypatch
    ):
        import event_stream

        redis_client = self._redis_backed(monkeypatch)

        first = client.post("/api/events/transaction", json=self._payload(), headers=auth_headers)
        second = client.post("/api/events/transaction", json=self._payload(), headers=auth_headers)

        assert second.status_code == 202
        assert second.json()["duplicate"] is True
        # Same verdict_id back, and only one message on the stream.
        assert second.json()["verdict_id"] == first.json()["verdict_id"]
        assert event_stream.stream_depth(redis_client)["length"] == 1

    def test_distinct_event_ids_are_both_accepted(self, client, auth_headers, monkeypatch):
        import event_stream

        redis_client = self._redis_backed(monkeypatch)

        client.post("/api/events/transaction", json=self._payload("evt-a"), headers=auth_headers)
        client.post("/api/events/transaction", json=self._payload("evt-b"), headers=auth_headers)

        assert event_stream.stream_depth(redis_client)["length"] == 2

    def test_returns_503_rather_than_pretending_to_queue_without_redis(self, client, auth_headers):
        # The app's default in tests is no Redis client at all.
        r = client.post("/api/events/transaction", json=self._payload(), headers=auth_headers)

        assert r.status_code == 503
        assert "requires Redis" in r.json()["detail"]

    def test_requires_an_api_key(self, client):
        r = client.post("/api/events/transaction", json=self._payload())
        assert r.status_code == 401

    def test_rejects_a_payload_with_no_event_id(self, client, auth_headers, monkeypatch):
        self._redis_backed(monkeypatch)

        r = client.post(
            "/api/events/transaction", json={"TransactionAmt": 100.0}, headers=auth_headers
        )

        assert r.status_code == 422

    def test_entity_id_is_optional(self, client, auth_headers, monkeypatch):
        self._redis_backed(monkeypatch)

        r = client.post(
            "/api/events/transaction",
            json={"event_id": "evt-no-entity", "TransactionAmt": 100.0},
            headers=auth_headers,
        )

        assert r.status_code == 202


class TestFeedbackExport:
    def _disposed_item(self, client, auth_headers, disposition="CONFIRMED_FRAUD"):
        """Score something that lands in the queue, then dispose it."""
        r = client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0},
                        headers=auth_headers)
        verdict_id = r.json()["verdict_id"]
        client.post(f"/api/review-queue/{verdict_id}/disposition",
                    json={"disposition": disposition}, headers=auth_headers)
        return verdict_id

    def test_exports_disposed_items_as_labelled_rows(self, client, auth_headers):
        verdict_id = self._disposed_item(client, auth_headers)

        body = client.get("/api/feedback/export", headers=auth_headers).json()

        assert body["count"] == 1
        assert body["confirmed_fraud"] == 1
        row = body["rows"][0]
        assert row["verdict_id"] == verdict_id
        assert row["isFraud"] == 1

    def test_a_false_positive_is_labelled_zero(self, client, auth_headers):
        self._disposed_item(client, auth_headers, "FALSE_POSITIVE")

        body = client.get("/api/feedback/export", headers=auth_headers).json()

        assert body["rows"][0]["isFraud"] == 0
        assert body["false_positive"] == 1

    def test_an_undisposed_item_is_not_a_label(self, client, auth_headers):
        client.post("/api/score", json={"entity_id": "entity-a", "txn_index": 0},
                    headers=auth_headers)

        assert client.get("/api/feedback/export", headers=auth_headers).json()["count"] == 0

    def test_the_response_carries_the_sampling_bias_caveat(self, client, auth_headers):
        """These labels exist only for flagged transactions. The caveat
        ships with the data rather than living only in a docstring."""
        body = client.get("/api/feedback/export", headers=auth_headers).json()

        assert "censored sample" in body["bias_warning"]

    def test_exported_rows_carry_the_features_the_model_saw(self, client, auth_headers):
        self._disposed_item(client, auth_headers)

        row = client.get("/api/feedback/export", headers=auth_headers).json()["rows"][0]

        # From the sample fixture's transaction.
        assert row["TransactionAmt"] == 100.0
        assert row["transaction_dt"] == 1000.0

    def test_requires_an_api_key(self, client):
        assert client.get("/api/feedback/export").status_code == 401


class TestEscalationAlerts:
    """Integration coverage for the wiring, not the transition/cooldown
    logic itself — that's tests/test_notifications.py. Swaps in a spy
    EscalationNotifier so a real scoring call through the API can be
    asserted to have reached it."""

    def _spy_notifier(self, monkeypatch):
        import api.main as main
        from notifications import EscalationNotifier

        sent = []
        monkeypatch.setattr(main, "_notifier", EscalationNotifier(sent.append))
        return sent

    def test_an_escalating_verdict_notifies(self, client, auth_headers, monkeypatch):
        import api.main as main
        from tests.conftest import FakeExplainer

        sent = self._spy_notifier(monkeypatch)
        monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer(risk_score=90.0))

        client.post(
            "/api/score-custom",
            json={"TransactionAmt": 100.0, "attach_to_entity_id": "entity-escalates"},
            headers=auth_headers,
        )

        assert len(sent) == 1
        assert sent[0]["entity_id"] == "entity-escalates"
        assert sent[0]["from_state"] == "NORMAL"
        assert sent[0]["to_state"] == "WATCH"

    def test_a_verdict_that_does_not_change_state_does_not_notify(self, client, auth_headers, monkeypatch):
        import api.main as main
        from tests.conftest import FakeExplainer

        sent = self._spy_notifier(monkeypatch)
        monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer(risk_score=5.0))

        client.post(
            "/api/score-custom",
            json={"TransactionAmt": 100.0, "attach_to_entity_id": "entity-calm"},
            headers=auth_headers,
        )

        assert sent == []

    def test_an_unattached_custom_transaction_never_notifies(self, client, auth_headers, monkeypatch):
        import api.main as main
        from tests.conftest import FakeExplainer

        sent = self._spy_notifier(monkeypatch)
        monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer(risk_score=90.0))

        client.post("/api/score-custom", json={"TransactionAmt": 100.0}, headers=auth_headers)

        assert sent == []


class TestDeadLetterEndpoint:
    def test_lists_dead_lettered_events(self, client, auth_headers, monkeypatch):
        import api.main as main
        import event_stream
        import fakeredis

        redis_client = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(main, "_redis_client", redis_client)
        event_stream.ensure_group(redis_client)
        message_id = event_stream.publish_event(redis_client, {"event_id": "evt-dead"})
        event_stream.read_events(redis_client, "c", block_ms=0)
        event_stream.dead_letter(redis_client, {"event_id": "evt-dead"}, "boom", message_id)

        r = client.get("/api/events/dead-letter", headers=auth_headers)

        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["event"]["event_id"] == "evt-dead"
        assert items[0]["error"] == "boom"

    def test_is_empty_when_nothing_has_failed(self, client, auth_headers, monkeypatch):
        import api.main as main
        import fakeredis

        monkeypatch.setattr(main, "_redis_client", fakeredis.FakeRedis(decode_responses=True))

        assert client.get("/api/events/dead-letter", headers=auth_headers).json()["items"] == []

    def test_returns_503_without_redis(self, client, auth_headers):
        assert client.get("/api/events/dead-letter", headers=auth_headers).status_code == 503

    def test_requires_an_api_key(self, client):
        assert client.get("/api/events/dead-letter").status_code == 401


class TestEscalationAblation:
    def test_returns_a_helpful_message_when_no_report_exists_yet(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main

        monkeypatch.setattr(main, "ESCALATION_ABLATION_REPORT_PATH", str(tmp_path / "nope.txt"))

        body = client.get("/api/escalation-ablation", headers=auth_headers).json()
        assert body["report"] is None
        assert body["summary"] is None
        assert "escalation_ablation.py" in body["message"]

    def test_serves_the_structured_summary_alongside_the_text_report(
        self, client, auth_headers, monkeypatch, tmp_path
    ):
        import api.main as main
        import json

        report_path = tmp_path / "report.txt"
        report_path.write_text("Escalation ablation study", encoding="utf-8")
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps({"n_transactions": 10, "baseline": {"recall": 0.5}}))
        monkeypatch.setattr(main, "ESCALATION_ABLATION_REPORT_PATH", str(report_path))
        monkeypatch.setattr(main, "ESCALATION_ABLATION_SUMMARY_PATH", str(summary_path))

        body = client.get("/api/escalation-ablation", headers=auth_headers).json()

        assert body["report"] == "Escalation ablation study"
        assert body["summary"]["n_transactions"] == 10

    def test_summary_is_null_for_a_report_generated_before_it_existed(
        self, client, auth_headers, monkeypatch, tmp_path
    ):
        import api.main as main

        report_path = tmp_path / "report.txt"
        report_path.write_text("Escalation ablation study", encoding="utf-8")
        monkeypatch.setattr(main, "ESCALATION_ABLATION_REPORT_PATH", str(report_path))
        monkeypatch.setattr(main, "ESCALATION_ABLATION_SUMMARY_PATH", str(tmp_path / "nope.json"))

        body = client.get("/api/escalation-ablation", headers=auth_headers).json()

        assert body["report"] == "Escalation ablation study"
        assert body["summary"] is None


class TestColdStartAnalysis:
    """Same file-read-or-fallback-message pattern as the other offline
    report endpoints (get_cost_sensitivity, get_escalation_ablation,
    get_drift_analysis, get_consistency_analysis) in api/main.py."""

    def test_returns_a_helpful_message_when_no_report_exists_yet(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main

        monkeypatch.setattr(main, "COLD_START_REPORT_PATH", str(tmp_path / "does-not-exist.txt"))

        r = client.get("/api/cold-start-analysis", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["report"] is None
        assert "graph_features_ablation.py" in body["message"]

    def test_returns_the_real_report_content_when_present(self, client, auth_headers, monkeypatch, tmp_path):
        import api.main as main

        report_path = tmp_path / "cold_start_report.txt"
        report_path.write_text("Cold-start ablation report\n===========================\n")
        monkeypatch.setattr(main, "COLD_START_REPORT_PATH", str(report_path))

        r = client.get("/api/cold-start-analysis", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["message"] is None
        assert "Cold-start ablation report" in body["report"]

    def test_requires_api_key(self, client):
        r = client.get("/api/cold-start-analysis")
        assert r.status_code == 401

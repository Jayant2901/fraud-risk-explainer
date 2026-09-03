"""
Route coverage for api/main.py using FastAPI's TestClient against the
fully-faked app from conftest.py (fake sample data, fake explainer, fake
LLM agent) — no trained model, no dataset, no GEMINI_API_KEY needed.
"""
from api.main import decide_action


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
        assert r.json() == {"status": "ok"}


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

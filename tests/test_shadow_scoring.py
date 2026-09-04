"""
ShadowComparison tests, run against both backends (in-process and Redis
via fakeredis) — the dual-implementation pattern test_circuit_breaker.py
established. ShadowScorer itself (a thin joblib-model wrapper) isn't
covered here — src/risk_explainer.py's near-identical scoring path
already is, and duplicating that against a real model file would just
slow the suite down for no new coverage.
"""
import fakeredis
import pytest

from shadow_scoring import ShadowComparison, create_shadow_scorer


@pytest.fixture(params=["in-process", "redis"])
def comparison_factory(request):
    redis_client = (
        fakeredis.FakeRedis(decode_responses=True) if request.param == "redis" else None
    )
    return lambda: ShadowComparison(redis_client)


class TestFactory:
    def test_no_shadow_model_path_yields_none(self, monkeypatch):
        monkeypatch.delenv("SHADOW_MODEL_PATH", raising=False)
        import shadow_scoring
        monkeypatch.setattr(shadow_scoring, "SHADOW_MODEL_PATH", None)

        assert create_shadow_scorer() is None


class TestRecordingAndSummary:
    def test_an_empty_comparison_reports_no_rate(self, comparison_factory):
        summary = comparison_factory().summary()

        assert summary == {
            "configured": True, "total_scored": 0, "agreement_rate": None, "action_pairs": [],
        }

    def test_agreement_is_counted_when_actions_match(self, comparison_factory):
        comparison = comparison_factory()
        comparison.record("REVIEW", "REVIEW")
        comparison.record("BLOCK", "BLOCK")

        summary = comparison.summary()
        assert summary["total_scored"] == 2
        assert summary["agreement_rate"] == 1.0

    def test_disagreement_lowers_the_rate_but_still_counts(self, comparison_factory):
        comparison = comparison_factory()
        comparison.record("REVIEW", "REVIEW")
        comparison.record("REVIEW", "BLOCK")

        summary = comparison.summary()
        assert summary["total_scored"] == 2
        assert summary["agreement_rate"] == 0.5

    def test_action_pairs_are_broken_out_by_live_and_shadow_action(self, comparison_factory):
        comparison = comparison_factory()
        comparison.record("REVIEW", "REVIEW")
        comparison.record("REVIEW", "REVIEW")
        comparison.record("ALLOW", "REVIEW")

        pairs = {(p["live_action"], p["shadow_action"]): p["count"] for p in comparison.summary()["action_pairs"]}
        assert pairs == {("REVIEW", "REVIEW"): 2, ("ALLOW", "REVIEW"): 1}

    def test_reset_clears_recorded_counts(self, comparison_factory):
        comparison = comparison_factory()
        comparison.record("REVIEW", "REVIEW")

        comparison.reset()

        assert comparison.summary()["total_scored"] == 0


class TestSharedState:
    def test_two_processes_on_one_redis_share_the_counts(self):
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        worker_a = ShadowComparison(redis_client)
        worker_b = ShadowComparison(redis_client)

        worker_a.record("REVIEW", "REVIEW")
        worker_b.record("BLOCK", "BLOCK")

        assert worker_a.summary()["total_scored"] == 2
        assert worker_b.summary()["total_scored"] == 2

    def test_in_process_comparisons_do_not_share_state(self):
        a = ShadowComparison()
        b = ShadowComparison()

        a.record("REVIEW", "REVIEW")

        assert a.summary()["total_scored"] == 1
        assert b.summary()["total_scored"] == 0

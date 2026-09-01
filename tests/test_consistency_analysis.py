"""
Tests the PURE computation functions only (boundary fragility, fallback
detection, modal-action selection, pair aggregation, escalation-context
builders) against small synthetic inputs — no real API key or network
access needed. run()/select_sample()'s real-API-calling path is a manual
script (like train_model.py's run()) and stays untested here, same as
the acceptance criteria for this module says it should.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from google.genai import errors

from llm_agent import RiskExplainerAgent, RiskVerdict
from consistency_analysis import (
    is_near_boundary,
    compute_boundary_fragility,
    is_fallback_response,
    modal_action,
    aggregate_pair,
    normal_escalation,
    elevated_escalation,
    MIN_VALID_RESPONSES,
)

TOP_FACTORS = [{"label": "transaction amount", "value": "500", "contribution": 0.5}]


def _client_error(code: int, message: str) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": message}})


def make_client(parsed=None, raises=None):
    client = MagicMock()
    if raises is not None:
        client.models.generate_content.side_effect = raises
    else:
        resp = MagicMock()
        resp.parsed = parsed
        client.models.generate_content.return_value = resp
    return client


class TestIsNearBoundary:
    def test_within_tolerance_of_40_is_near(self):
        assert is_near_boundary(41.5)
        assert is_near_boundary(38.0)

    def test_within_tolerance_of_80_is_near(self):
        assert is_near_boundary(81.9)

    def test_far_from_either_boundary_is_not_near(self):
        assert not is_near_boundary(60.0)
        assert not is_near_boundary(10.0)

    def test_exactly_at_tolerance_edge_counts_as_near(self):
        assert is_near_boundary(42.0)  # exactly 40 + 2
        assert not is_near_boundary(42.1)


class TestComputeBoundaryFragility:
    def test_hand_computable_fraction(self):
        # 5 flagged scores: 41 (near 40), 79 (near 80), 60 (far), 100 (far), 38.5 (near 40)
        scores = [41.0, 79.0, 60.0, 100.0, 38.5]
        result = compute_boundary_fragility(scores)
        assert result["n_flagged"] == 5
        assert result["n_near_boundary"] == 3
        assert result["fraction_near_boundary"] == pytest.approx(0.6)

    def test_empty_input_does_not_divide_by_zero(self):
        result = compute_boundary_fragility([])
        assert result["n_flagged"] == 0
        assert result["fraction_near_boundary"] == 0.0

    def test_no_flagged_scores_near_a_boundary(self):
        result = compute_boundary_fragility([10.0, 60.0, 95.0])
        assert result["n_near_boundary"] == 0
        assert result["fraction_near_boundary"] == 0.0


class TestIsFallbackResponse:
    """Uses a REAL RiskExplainerAgent with a MagicMock Gemini client (same
    approach as test_llm_agent.py) so this proves is_fallback_response()
    correctly recognizes the actual dicts llm_agent.py produces, not just
    a hand-constructed stand-in."""

    def test_detects_a_real_rate_limited_fallback(self):
        agent = RiskExplainerAgent(client=make_client(raises=_client_error(429, "slow down")))
        verdict = agent.explain(55, TOP_FACTORS, None)
        assert is_fallback_response(verdict)

    def test_detects_a_real_unauthenticated_fallback(self):
        agent = RiskExplainerAgent(client=make_client(raises=_client_error(401, "bad key")))
        verdict = agent.explain(55, TOP_FACTORS, None)
        assert is_fallback_response(verdict)

    def test_real_successful_response_is_not_flagged_as_fallback(self):
        parsed = RiskVerdict(explanation="fine", action="ALLOW", escalated_due_to_history=False, rationale="low risk")
        agent = RiskExplainerAgent(client=make_client(parsed=parsed))
        verdict = agent.explain(10, TOP_FACTORS, None)
        assert not is_fallback_response(verdict)

    def test_missing_rationale_key_is_not_a_crash(self):
        assert is_fallback_response({}) is False


class TestModalAction:
    def test_clear_majority(self):
        assert modal_action(["BLOCK", "BLOCK", "REVIEW"]) == "BLOCK"

    def test_unanimous(self):
        assert modal_action(["ALLOW", "ALLOW", "ALLOW"]) == "ALLOW"

    def test_tie_breaks_by_first_appearance_in_call_order(self):
        assert modal_action(["REVIEW", "BLOCK", "REVIEW", "BLOCK"]) == "REVIEW"
        assert modal_action(["BLOCK", "REVIEW", "BLOCK", "REVIEW"]) == "BLOCK"

    def test_single_response(self):
        assert modal_action(["ALLOW"]) == "ALLOW"


class TestAggregatePair:
    def test_all_valid_and_unanimous(self):
        verdicts = [{"action": "BLOCK", "rationale": "high score"}] * 5
        result = aggregate_pair(verdicts, deterministic_action="BLOCK")
        assert result["status"] == "ok"
        assert result["n_valid"] == 5
        assert result["n_excluded_fallback"] == 0
        assert result["modal_action"] == "BLOCK"
        assert result["self_consistency_rate"] == 1.0
        assert result["cross_agreement"] is True

    def test_excludes_fallback_responses_from_the_count(self):
        # 3 real BLOCK responses + 2 rate-limited fallbacks reported as
        # REVIEW -- the fallbacks must NOT count as real disagreement.
        verdicts = (
            [{"action": "BLOCK", "rationale": "high score"}] * 3
            + [{"action": "REVIEW", "rationale": "Falling back to manual review — rate-limited"}] * 2
        )
        result = aggregate_pair(verdicts, deterministic_action="BLOCK")
        assert result["status"] == "ok"
        assert result["n_valid"] == 3
        assert result["n_excluded_fallback"] == 2
        assert result["modal_action"] == "BLOCK"
        assert result["self_consistency_rate"] == 1.0  # NOT 3/5=0.6

    def test_insufficient_valid_data_is_reported_not_fabricated(self):
        verdicts = (
            [{"action": "REVIEW", "rationale": "Falling back to manual review — rate-limited"}] * 4
            + [{"action": "BLOCK", "rationale": "real"}]
        )
        result = aggregate_pair(verdicts, deterministic_action="BLOCK", min_valid=2)
        assert result["status"] == "insufficient_data"
        assert result["n_valid"] == 1
        assert result["n_excluded_fallback"] == 4
        assert result["modal_action"] is None
        assert result["self_consistency_rate"] is None
        assert result["cross_agreement"] is None

    def test_default_min_valid_matches_module_constant(self):
        verdicts = [{"action": "BLOCK", "rationale": "real"}] * (MIN_VALID_RESPONSES - 1)
        result = aggregate_pair(verdicts, deterministic_action="BLOCK")
        assert result["status"] == "insufficient_data"

    def test_cross_agreement_false_when_modal_differs_from_deterministic(self):
        verdicts = [{"action": "REVIEW", "rationale": "moderate"}] * 5
        result = aggregate_pair(verdicts, deterministic_action="BLOCK")
        assert result["cross_agreement"] is False

    def test_disagreement_lowers_self_consistency_rate(self):
        verdicts = (
            [{"action": "REVIEW", "rationale": "x"}] * 3
            + [{"action": "BLOCK", "rationale": "y"}] * 2
        )
        result = aggregate_pair(verdicts, deterministic_action="REVIEW")
        assert result["modal_action"] == "REVIEW"
        assert result["self_consistency_rate"] == pytest.approx(0.6)


class TestEscalationContextBuilders:
    def test_normal_escalation_has_no_history(self):
        esc = normal_escalation("entity-1")
        assert esc["state"] == "NORMAL"
        assert esc["recent_verdict_count"] == 0

    def test_elevated_escalation_actually_reaches_elevated_state(self):
        esc = elevated_escalation("entity-1", risk_score=90.0)
        assert esc["state"] == "ELEVATED"
        assert esc["recent_risky_count"] >= 1

    def test_different_entities_do_not_share_computed_state(self):
        esc_a = elevated_escalation("entity-a", risk_score=90.0)
        esc_b = normal_escalation("entity-b")
        assert esc_a["state"] == "ELEVATED"
        assert esc_b["state"] == "NORMAL"

"""
Cost-based threshold analysis.

Raw precision/recall don't tell a risk team what threshold to actually
operate at — that requires knowing the RELATIVE COST of the two error
types:

  - False Negative (missed fraud): the platform eats the fraud loss.
    Assume ~avg_fraud_loss per missed fraudulent transaction.
  - False Positive (wrongly blocked/reviewed legitimate transaction):
    costs customer trust/friction, not direct loss, but has a real
    (smaller, harder to measure) cost — assume ~avg_fp_cost per
    wrongly-flagged legitimate transaction (support burden, churn risk,
    lost transaction revenue).

These two assumed constants are configurable — a real deployment would
pull them from finance/ops data. Here they're clearly-labeled
assumptions so the tradeoff logic is transparent, not hidden inside a
single "accuracy" number.

The output is a cost curve across thresholds, letting you pick the
threshold that MINIMIZES TOTAL EXPECTED COST rather than one that
maximizes a threshold-agnostic metric like AUC.
"""
import numpy as np
import pandas as pd

# Assumed business costs (INR). Documented, adjustable placeholders —
# swap these for real figures if you have them (e.g. from a Razorpay
# ops/finance breakdown).
DEFAULT_AVG_FRAUD_LOSS = 5000.0     # cost of a missed fraudulent transaction
DEFAULT_AVG_FP_COST = 150.0        # cost of wrongly flagging a legitimate transaction

# threshold_sensitivity()'s default grid: 0.5x-2x the defaults above, on
# both axes — a plausible range for how wrong a single point-estimate
# guess at these two costs could be.
DEFAULT_SENSITIVITY_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]


def cost_curve(y_true, y_proba, thresholds=None,
                avg_fraud_loss: float = DEFAULT_AVG_FRAUD_LOSS,
                avg_fp_cost: float = DEFAULT_AVG_FP_COST) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per threshold:
    threshold, false_negatives, false_positives, total_cost,
    precision, recall.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    n_fraud = y_true.sum()
    n_legit = len(y_true) - n_fraud

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())

        total_cost = fn * avg_fraud_loss + fp * avg_fp_cost
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_fraud if n_fraud > 0 else 0.0

        rows.append({
            "threshold": round(float(t), 3),
            "false_negatives": fn,
            "false_positives": fp,
            "true_positives": tp,
            "total_cost": round(total_cost, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })

    return pd.DataFrame(rows)


def optimal_threshold(y_true, y_proba,
                       avg_fraud_loss: float = DEFAULT_AVG_FRAUD_LOSS,
                       avg_fp_cost: float = DEFAULT_AVG_FP_COST) -> dict:
    """
    Returns the threshold that minimizes total expected cost, plus its
    stats, plus a comparison against the naive 0.5 threshold so the
    pitch can show "cost-optimal beats default by X%."
    """
    curve = cost_curve(y_true, y_proba, avg_fraud_loss=avg_fraud_loss, avg_fp_cost=avg_fp_cost)
    best_row = curve.loc[curve["total_cost"].idxmin()]

    default_row = curve.iloc[(curve["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]

    savings = default_row["total_cost"] - best_row["total_cost"]
    savings_pct = (savings / default_row["total_cost"] * 100) if default_row["total_cost"] > 0 else 0.0

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_total_cost": float(best_row["total_cost"]),
        "optimal_precision": float(best_row["precision"]),
        "optimal_recall": float(best_row["recall"]),
        "default_threshold_cost": float(default_row["total_cost"]),
        "estimated_savings": float(savings),
        "estimated_savings_pct": round(float(savings_pct), 1),
        "curve": curve,
    }


def threshold_sensitivity(y_true, y_proba,
                           fraud_loss_multipliers: list | None = None,
                           fp_cost_multipliers: list | None = None,
                           base_fraud_loss: float = DEFAULT_AVG_FRAUD_LOSS,
                           base_fp_cost: float = DEFAULT_AVG_FP_COST) -> dict:
    """
    The default cost assumptions (avg_fraud_loss=Rs 5,000, avg_fp_cost=
    Rs 150) are a single, undefended point estimate. This sweeps a grid
    of both costs (multiples of the defaults, e.g. 0.5x-2x) and reports
    how the COST-OPTIMAL THRESHOLD and expected savings shift across
    that grid — so a real risk team can see how sensitive "the" optimal
    threshold actually is to getting these two numbers wrong, instead of
    trusting one pair blindly.
    """
    if fraud_loss_multipliers is None:
        fraud_loss_multipliers = DEFAULT_SENSITIVITY_MULTIPLIERS
    if fp_cost_multipliers is None:
        fp_cost_multipliers = DEFAULT_SENSITIVITY_MULTIPLIERS

    grid = []
    for fl_mult in fraud_loss_multipliers:
        for fp_mult in fp_cost_multipliers:
            avg_fraud_loss = base_fraud_loss * fl_mult
            avg_fp_cost = base_fp_cost * fp_mult
            result = optimal_threshold(y_true, y_proba, avg_fraud_loss=avg_fraud_loss, avg_fp_cost=avg_fp_cost)
            grid.append({
                "fraud_loss_multiplier": fl_mult,
                "fp_cost_multiplier": fp_mult,
                "avg_fraud_loss": avg_fraud_loss,
                "avg_fp_cost": avg_fp_cost,
                "optimal_threshold": result["optimal_threshold"],
                "optimal_total_cost": result["optimal_total_cost"],
                "estimated_savings_pct": result["estimated_savings_pct"],
            })

    return {
        "base_fraud_loss": base_fraud_loss,
        "base_fp_cost": base_fp_cost,
        "fraud_loss_multipliers": fraud_loss_multipliers,
        "fp_cost_multipliers": fp_cost_multipliers,
        "grid": grid,
    }

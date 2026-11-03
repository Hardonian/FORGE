"""Active learning loop and fast surrogate screening with uncertainty acquisition."""
from __future__ import annotations
from typing import Dict, List, Tuple
from forge.core.models import DuctSpec, PhysicsConditions
from forge.surrogate.model import DuctFlowSurrogate


def screen_candidates(
    candidates: List[DuctSpec],
    conditions: PhysicsConditions,
    model: DuctFlowSurrogate,
    target_max_dp_pa: float = 80.0,
) -> List[Tuple[DuctSpec, float]]:
    """Screen candidates using the neural surrogate, ranking those meeting flow constraints."""
    passed: List[Tuple[DuctSpec, float]] = []
    for spec in candidates:
        pred = model.predict_spec(spec, conditions)
        dp = pred["predicted_delta_p_pa"]
        if dp <= target_max_dp_pa:
            passed.append((spec, dp))

    passed.sort(key=lambda x: x[1])
    return passed


def rank_candidates_by_acquisition(
    candidates: List[DuctSpec],
    conditions: PhysicsConditions,
    model: DuctFlowSurrogate,
    exploration_kappa: float = 2.0,
) -> List[Dict[str, object]]:
    """Rank candidates using Lower Confidence Bound (LCB) acquisition: mu_dP - kappa * sigma_dP.

    Prioritizes candidates with either very low predicted pressure drop or high epistemic uncertainty
    for subsequent ground-truth CFD verification.
    """
    scored: List[Dict[str, object]] = []
    for spec in candidates:
        unc = model.predict_with_uncertainty(spec, conditions)
        mu_dp = unc["predicted_delta_p_pa"]
        std_dp = unc["std_delta_p_pa"]
        # Lower LCB score is better (minimizing pressure drop while rewarding high uncertainty)
        lcb_score = mu_dp - exploration_kappa * std_dp
        scored.append({
            "spec": spec,
            "predicted_delta_p_pa": mu_dp,
            "std_delta_p_pa": std_dp,
            "lcb_acquisition_score": round(lcb_score, 3),
        })

    scored.sort(key=lambda item: float(item["lcb_acquisition_score"]))
    return scored

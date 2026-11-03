"""Active learning loop and fast surrogate screening."""
from __future__ import annotations
from typing import List, Tuple
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

    # Sort by lowest predicted pressure drop
    passed.sort(key=lambda x: x[1])
    return passed


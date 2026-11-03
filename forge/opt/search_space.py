"""Design search space parameterization and Latin Hypercube sampling."""
from __future__ import annotations
import random
from typing import List
from pydantic import BaseModel, Field
from forge.core.models import DuctSpec, DuctMaterial


class ParamRange(BaseModel):
    min_val: float
    max_val: float


class DuctSearchSpace(BaseModel):
    """Bounds and constraints for generative design optimization."""
    inlet_width: float = 120.0
    inlet_height: float = 120.0
    outlet_width: float = 140.0
    outlet_height: float = 45.0
    wall: float = 2.0

    length_range: ParamRange = Field(default_factory=lambda: ParamRange(min_val=60.0, max_val=160.0))
    offset_x_range: ParamRange = Field(default_factory=lambda: ParamRange(min_val=-30.0, max_val=30.0))
    offset_y_range: ParamRange = Field(default_factory=lambda: ParamRange(min_val=-20.0, max_val=20.0))
    corner_radius_range: ParamRange = Field(default_factory=lambda: ParamRange(min_val=0.0, max_val=15.0))
    guide_vanes_options: List[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    material: DuctMaterial = DuctMaterial.PC_CF


def sample_latin_hypercube(space: DuctSearchSpace, num_samples: int = 50, seed: int = 42) -> List[DuctSpec]:
    """Sample candidates across the search space using Latin Hypercube Sampling (LHS)."""
    rng = random.Random(seed)

    # Divide intervals into N stratified bins
    def stratified_sample(p_range: ParamRange, n: int) -> List[float]:
        step = (p_range.max_val - p_range.min_val) / float(n)
        vals = [p_range.min_val + (i + rng.random()) * step for i in range(n)]
        rng.shuffle(vals)
        return vals

    lengths = stratified_sample(space.length_range, num_samples)
    offsets_x = stratified_sample(space.offset_x_range, num_samples)
    offsets_y = stratified_sample(space.offset_y_range, num_samples)
    radii = stratified_sample(space.corner_radius_range, num_samples)

    candidates: List[DuctSpec] = []
    for i in range(num_samples):
        vanes = rng.choice(space.guide_vanes_options)
        spec = DuctSpec(
            inlet_width=space.inlet_width,
            inlet_height=space.inlet_height,
            outlet_width=space.outlet_width,
            outlet_height=space.outlet_height,
            length=round(lengths[i], 1),
            wall=space.wall,
            offset_x=round(offsets_x[i], 1),
            offset_y=round(offsets_y[i], 1),
            corner_radius=round(radii[i], 1),
            guide_vanes=vanes,
            material=space.material,
        )
        candidates.append(spec)

    return candidates

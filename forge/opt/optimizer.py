"""Multi-objective NSGA-II Pareto optimizer for inverse aerodynamic design."""
from __future__ import annotations
import random
from typing import List, NamedTuple
from forge.core.models import DuctSpec, PhysicsConditions, PhysicsMetrics
from forge.core.geometry import calculate_mesh_volume
from forge.cad.mesh import build_duct_mesh
from forge.physics.solver_1d import solve_duct_physics
from forge.opt.search_space import DuctSearchSpace, sample_latin_hypercube


class CandidateEvaluation(NamedTuple):
    spec: DuctSpec
    physics: PhysicsMetrics
    mass_grams: float
    # Objective values (all to be minimized):
    # f1: pressure drop Pa
    # f2: 1.0 - flow uniformity (minimizing this maximizes uniformity)
    # f3: mass in grams
    obj_pressure_pa: float
    obj_non_uniformity: float
    obj_mass_g: float
    pareto_rank: int = 0
    crowding_dist: float = 0.0


def evaluate_spec(spec: DuctSpec, conditions: PhysicsConditions) -> CandidateEvaluation:
    """Evaluate physics and manufacturing mass for a given duct specification."""
    phys = solve_duct_physics(spec, conditions)
    tris = build_duct_mesh(spec)
    vol_cm3 = abs(calculate_mesh_volume(tris)) / 1000.0
    mass_g = vol_cm3 * 1.25  # standard filament density

    return CandidateEvaluation(
        spec=spec,
        physics=phys,
        mass_grams=round(mass_g, 2),
        obj_pressure_pa=phys.total_pressure_drop_pa,
        obj_non_uniformity=1.0 - phys.flow_uniformity_index,
        obj_mass_g=round(mass_g, 2),
    )


def fast_non_dominated_sort(evals: List[CandidateEvaluation]) -> List[List[int]]:
    """Fast non-dominated sorting algorithm for NSGA-II."""
    n = len(evals)
    domination_sets: List[List[int]] = [[] for _ in range(n)]
    domination_counts = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            # Check if p dominates q
            p_better = (
                evals[p].obj_pressure_pa <= evals[q].obj_pressure_pa
                and evals[p].obj_non_uniformity <= evals[q].obj_non_uniformity
                and evals[p].obj_mass_g <= evals[q].obj_mass_g
            )
            p_strictly_better = (
                evals[p].obj_pressure_pa < evals[q].obj_pressure_pa
                or evals[p].obj_non_uniformity < evals[q].obj_non_uniformity
                or evals[p].obj_mass_g < evals[q].obj_mass_g
            )

            q_better = (
                evals[q].obj_pressure_pa <= evals[p].obj_pressure_pa
                and evals[q].obj_non_uniformity <= evals[p].obj_non_uniformity
                and evals[q].obj_mass_g <= evals[p].obj_mass_g
            )
            q_strictly_better = (
                evals[q].obj_pressure_pa < evals[p].obj_pressure_pa
                or evals[q].obj_non_uniformity < evals[p].obj_non_uniformity
                or evals[q].obj_mass_g < evals[p].obj_mass_g
            )

            if p_better and p_strictly_better:
                domination_sets[p].append(q)
                domination_counts[q] += 1
            elif q_better and q_strictly_better:
                domination_sets[q].append(p)
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and len(fronts[i]) > 0:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def run_nsga2_optimization(
    space: DuctSearchSpace,
    conditions: PhysicsConditions,
    population_size: int = 30,
    generations: int = 5,
    seed: int = 42,
) -> List[CandidateEvaluation]:
    """Execute NSGA-II multi-objective optimization returning the Pareto-optimal front."""
    rng = random.Random(seed)
    # 1. Initialize population via Latin Hypercube Sampling
    population_specs = sample_latin_hypercube(space, population_size, seed=seed)
    population = [evaluate_spec(spec, conditions) for spec in population_specs]

    for gen in range(generations):
        # Generate offspring via crossover and mutation
        offspring_specs: List[DuctSpec] = []
        while len(offspring_specs) < population_size:
            p1 = rng.choice(population).spec
            p2 = rng.choice(population).spec

            # Simulated Binary Crossover (SBX)
            alpha = rng.random()
            new_length = alpha * p1.length + (1.0 - alpha) * p2.length
            new_ox = alpha * p1.offset_x + (1.0 - alpha) * p2.offset_x
            new_oy = alpha * p1.offset_y + (1.0 - alpha) * p2.offset_y
            new_cr = alpha * p1.corner_radius + (1.0 - alpha) * p2.corner_radius
            new_vanes = rng.choice([p1.guide_vanes, p2.guide_vanes])

            # Mutation
            if rng.random() < 0.2:
                new_length = max(space.length_range.min_val, min(space.length_range.max_val, new_length + rng.uniform(-10, 10)))
            if rng.random() < 0.2:
                new_ox = max(space.offset_x_range.min_val, min(space.offset_x_range.max_val, new_ox + rng.uniform(-5, 5)))
            if rng.random() < 0.2:
                new_vanes = rng.choice(space.guide_vanes_options)

            try:
                child = DuctSpec(
                    inlet_width=space.inlet_width,
                    inlet_height=space.inlet_height,
                    outlet_width=space.outlet_width,
                    outlet_height=space.outlet_height,
                    length=round(new_length, 1),
                    wall=space.wall,
                    offset_x=round(new_ox, 1),
                    offset_y=round(new_oy, 1),
                    corner_radius=round(new_cr, 1),
                    guide_vanes=new_vanes,
                    material=space.material,
                )
                offspring_specs.append(child)
            except ValueError:
                continue

        offspring = [evaluate_spec(s, conditions) for s in offspring_specs]
        combined = population + offspring

        # Non-dominated sort
        fronts = fast_non_dominated_sort(combined)
        new_pop: List[CandidateEvaluation] = []
        for front in fronts:
            if len(new_pop) + len(front) <= population_size:
                new_pop.extend([combined[idx] for idx in front])
            else:
                remaining = population_size - len(new_pop)
                new_pop.extend([combined[idx] for idx in front[:remaining]])
                break
        population = new_pop

    # Final non-dominated sort to extract Pareto Front (Rank 0)
    final_fronts = fast_non_dominated_sort(population)
    pareto_candidates = [population[i] for i in final_fronts[0]]
    # Sort by pressure drop ascending
    pareto_candidates.sort(key=lambda c: c.obj_pressure_pa)
    return pareto_candidates

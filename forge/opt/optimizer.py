"""Multi-objective NSGA-II Pareto optimizer with crowding-distance elitism."""
from __future__ import annotations
import random
from typing import Dict, List, NamedTuple
from forge.core.models import DuctSpec, MATERIAL_PROPERTIES, PhysicsConditions, PhysicsMetrics
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
    """Evaluate physics and material-specific manufacturing mass for a given duct specification."""
    phys = solve_duct_physics(spec, conditions)
    tris = build_duct_mesh(spec)
    vol_cm3 = abs(calculate_mesh_volume(tris)) / 1000.0
    mat_density = MATERIAL_PROPERTIES.get(spec.material.value, {}).get("density", 1.25)
    mass_g = vol_cm3 * mat_density

    return CandidateEvaluation(
        spec=spec,
        physics=phys,
        mass_grams=round(mass_g, 2),
        obj_pressure_pa=phys.total_pressure_drop_pa,
        obj_non_uniformity=round(1.0 - phys.flow_uniformity_index, 4),
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


def compute_crowding_distances(evals: List[CandidateEvaluation], front_indices: List[int]) -> Dict[int, float]:
    """Calculate NSGA-II crowding distance for indices in a Pareto front."""
    k = len(front_indices)
    if k == 0:
        return {}
    if k <= 2:
        return {idx: float("inf") for idx in front_indices}

    distances: Dict[int, float] = {idx: 0.0 for idx in front_indices}
    obj_getters = [
        lambda c: c.obj_pressure_pa,
        lambda c: c.obj_non_uniformity,
        lambda c: c.obj_mass_g,
    ]

    for get_obj in obj_getters:
        sorted_front = sorted(front_indices, key=lambda idx: get_obj(evals[idx]))
        f_min = get_obj(evals[sorted_front[0]])
        f_max = get_obj(evals[sorted_front[-1]])
        span = f_max - f_min
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")
        if span > 1e-9:
            for i in range(1, k - 1):
                prev_v = get_obj(evals[sorted_front[i - 1]])
                next_v = get_obj(evals[sorted_front[i + 1]])
                distances[sorted_front[i]] += (next_v - prev_v) / span

    return distances


def _annotate_population(evals: List[CandidateEvaluation]) -> List[CandidateEvaluation]:
    """Assign pareto_rank and crowding_dist to all evaluations."""
    fronts = fast_non_dominated_sort(evals)
    annotated = list(evals)
    for rank, front in enumerate(fronts):
        c_dists = compute_crowding_distances(evals, front)
        for idx in front:
            c = evals[idx]
            annotated[idx] = c._replace(pareto_rank=rank, crowding_dist=c_dists[idx])
    return annotated


def _tournament_select(pop: List[CandidateEvaluation], rng: random.Random) -> CandidateEvaluation:
    """Binary tournament selection using NSGA-II crowded-comparison operator."""
    a = rng.choice(pop)
    b = rng.choice(pop)
    if a.pareto_rank < b.pareto_rank:
        return a
    if b.pareto_rank < a.pareto_rank:
        return b
    return a if a.crowding_dist >= b.crowding_dist else b


def run_nsga2_optimization(
    space: DuctSearchSpace,
    conditions: PhysicsConditions,
    population_size: int = 30,
    generations: int = 5,
    seed: int = 42,
) -> List[CandidateEvaluation]:
    """Execute NSGA-II multi-objective optimization returning the Pareto-optimal front."""
    rng = random.Random(seed)
    population_specs = sample_latin_hypercube(space, population_size, seed=seed)
    population = _annotate_population([evaluate_spec(spec, conditions) for spec in population_specs])

    for _ in range(generations):
        offspring_specs: List[DuctSpec] = []
        while len(offspring_specs) < population_size:
            p1 = _tournament_select(population, rng).spec
            p2 = _tournament_select(population, rng).spec

            # Simulated Binary Crossover (SBX)
            alpha = rng.random()
            new_length = alpha * p1.length + (1.0 - alpha) * p2.length
            new_ox = alpha * p1.offset_x + (1.0 - alpha) * p2.offset_x
            new_oy = alpha * p1.offset_y + (1.0 - alpha) * p2.offset_y
            new_cr = alpha * p1.corner_radius + (1.0 - alpha) * p2.corner_radius
            new_vanes = rng.choice([p1.guide_vanes, p2.guide_vanes])

            # Polynomial / bounded Gaussian mutation across all genes
            if rng.random() < 0.25:
                new_length = max(space.length_range.min_val, min(space.length_range.max_val, new_length + rng.uniform(-12, 12)))
            if rng.random() < 0.25:
                new_ox = max(space.offset_x_range.min_val, min(space.offset_x_range.max_val, new_ox + rng.uniform(-6, 6)))
            if rng.random() < 0.25:
                new_oy = max(space.offset_y_range.min_val, min(space.offset_y_range.max_val, new_oy + rng.uniform(-6, 6)))
            if rng.random() < 0.25:
                new_cr = max(space.corner_radius_range.min_val, min(space.corner_radius_range.max_val, new_cr + rng.uniform(-3, 3)))
            if rng.random() < 0.25:
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

        # Elitist non-dominated sorting with crowding distance truncation
        fronts = fast_non_dominated_sort(combined)
        new_pop: List[CandidateEvaluation] = []
        for rank, front in enumerate(fronts):
            c_dists = compute_crowding_distances(combined, front)
            ranked_front = [
                combined[idx]._replace(pareto_rank=rank, crowding_dist=c_dists[idx])
                for idx in front
            ]
            if len(new_pop) + len(ranked_front) <= population_size:
                new_pop.extend(ranked_front)
            else:
                # Sort splitting front by crowding distance descending to preserve frontier diversity
                ranked_front.sort(key=lambda c: c.crowding_dist, reverse=True)
                remaining = population_size - len(new_pop)
                new_pop.extend(ranked_front[:remaining])
                break
        population = new_pop

    final_fronts = fast_non_dominated_sort(population)
    pareto_dists = compute_crowding_distances(population, final_fronts[0])
    pareto_candidates = [
        population[i]._replace(pareto_rank=0, crowding_dist=pareto_dists[i])
        for i in final_fronts[0]
    ]
    pareto_candidates.sort(key=lambda c: c.obj_pressure_pa)
    return pareto_candidates

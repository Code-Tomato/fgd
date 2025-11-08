"""FGD scheduler with tie-breakers and placement decisions.

This module provides the main scheduling logic that wraps the core FGD algorithm
with additional features like tie-breakers and rich placement metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Import core FGD algorithm
from .fgd import (
    ClusterState,
    Job,
    Model,
    compute_fragmentation as _compute_fragmentation_core,
    fgd_decide_placement as _fgd_decide_placement_core,
    _apply_allocation,
    _enumerate_candidates,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Configuration for FGD scheduler."""
    epsilon: float = 1e-9
    epsilon_int: float = 1e-6
    g_max: int = 200  # Maximum number of GPUs for dynamic planning
    tie_breakers: Sequence[str] = field(
        default_factory=lambda: [
            "keep_whole",
            "keep_max_partial",
            "keep_variance_low",
        ]
    )


@dataclass
class PlacementDecision:
    """Placement decision with metadata needed for tie-breakers."""
    gpu_indices: Optional[Tuple[int, ...]]
    delta_score: float
    gpus_leftover_after: List[float]


# ---------------------------------------------------------------------------
# Fragmentation computation
# ---------------------------------------------------------------------------


def compute_fragmentation(
    cluster: ClusterState, workload: Sequence[Model], jobs: Sequence[Job], config: Config
) -> float:
    """Compute fragmentation rate for the cluster.
    
    Wraps the core compute_fragmentation() to return the fragmentation rate.
    """
    from .fgd import Config as CoreConfig
    
    # Create minimal config for core function
    core_config = CoreConfig(epsilon=config.epsilon, epsilon_int=config.epsilon_int)
    
    # Get core fragmentation rate
    return _compute_fragmentation_core(cluster, workload, jobs, core_config)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def fgd_decide_placement(
    job: Job, cluster: ClusterState, workload: Sequence[Model], jobs: Sequence[Job], config: Config
) -> PlacementDecision:
    """Run Fragmentation Gradient Descent for a single job.
    
    Wraps the core fgd_decide_placement() to return PlacementDecision with
    full metadata and tie-breaker support.
    """
    from .fgd import Config as CoreConfig
    
    # Create minimal config for core function
    core_config = CoreConfig(epsilon=config.epsilon, epsilon_int=config.epsilon_int)
    
    gpus_before = cluster.gpus.copy()
    frag_before = compute_fragmentation(cluster, workload, jobs, config)
    
    # Get core placement decision
    best_indices = _fgd_decide_placement_core(job, cluster, workload, jobs, core_config)
    
    if best_indices is None:
        raise ValueError(f"Job with model {job.model_name} is infeasible for current cluster state")

    # If we have tie-breakers enabled, we need to check all candidates with equal deltas
    # Otherwise, just use the core result
    candidates = list(_enumerate_candidates(job, gpus_before, core_config))
    
    # Find all candidates with the same minimal delta
    best_delta = None
    best_candidates = []
    
    for candidate in candidates:
        simulated = ClusterState(gpus_before.copy())
        _apply_allocation(simulated.gpus, job, candidate, core_config)
        frag_after = compute_fragmentation(simulated, workload, jobs, config)
        delta = frag_after - frag_before
        
        if best_delta is None or delta < best_delta - config.epsilon:
            best_delta = delta
            best_candidates = [candidate]
        elif abs(delta - best_delta) <= config.epsilon:
            best_candidates.append(candidate)
    
    # If we have multiple candidates with same delta, apply tie-breakers
    if len(best_candidates) > 1 and config.tie_breakers:
        best_candidate = None
        best_decision = None
        
        for candidate in best_candidates:
            simulated = ClusterState(gpus_before.copy())
            _apply_allocation(simulated.gpus, job, candidate, core_config)
            frag_after = compute_fragmentation(simulated, workload, jobs, config)
            delta = frag_after - frag_before
            
            current = PlacementDecision(
                gpu_indices=candidate,
                delta_score=delta,
                gpus_leftover_after=simulated.gpus,
            )
            
            if best_decision is None or _is_candidate_better(current, best_decision, config):
                best_decision = current
                best_candidate = candidate
        
        best_indices = best_candidate
    
    # Build final PlacementDecision
    simulated = ClusterState(gpus_before.copy())
    _apply_allocation(simulated.gpus, job, best_indices, core_config)
    frag_after = compute_fragmentation(simulated, workload, jobs, config)
    delta = frag_after - frag_before
    
    return PlacementDecision(
        gpu_indices=best_indices,
        delta_score=delta,
        gpus_leftover_after=simulated.gpus,
    )


def commit_placement(job: Job, decision: PlacementDecision, cluster: ClusterState, config: Config) -> None:
    """Apply the chosen allocation to the real cluster state."""
    from .fgd import Config as CoreConfig
    
    core_config = CoreConfig(epsilon=config.epsilon, epsilon_int=config.epsilon_int)
    gpus = cluster.gpus
    _apply_allocation(gpus, job, decision.gpu_indices, core_config)


def schedule(
    queue: Sequence[Job],
    cluster: ClusterState,
    workload: Sequence[Model],
    config: Config,
) -> Dict:
    """FCFS scheduling loop that records placements."""
    from .output import _format_indices
    
    model_lookup = {model.model_name: model for model in workload}

    placements = []

    for t, job in enumerate(queue):
        if job.model_name not in model_lookup:
            raise ValueError(f"Job at index {t} references unknown model {job.model_name}")

        decision = fgd_decide_placement(job, cluster, workload, queue, config)
        commit_placement(job, decision, cluster, config)

        chosen = _format_indices(decision.gpu_indices)

        placements.append(
            {
                "t": t,
                "model_name": job.model_name,
                "gpu_demand": job.gpu_demand,
                "chosen_gpu_index": chosen,
            }
        )

    return {
        "placements": placements,
    }


# ---------------------------------------------------------------------------
# Tie-break helpers
# ---------------------------------------------------------------------------


def _is_candidate_better(current: PlacementDecision, best: PlacementDecision, config: Config) -> bool:
    """Compare two placement candidates using tie-breaker rules."""
    epsilon = config.epsilon

    if current.delta_score + epsilon < best.delta_score:
        return True
    if current.delta_score - epsilon > best.delta_score:
        return False

    for rule in config.tie_breakers:
        if rule == "keep_whole":
            diff = _count_whole_gpus(current.gpus_leftover_after, epsilon) - _count_whole_gpus(best.gpus_leftover_after, epsilon)
            if diff != 0:
                return diff > 0
        elif rule == "keep_max_partial":
            diff = _largest_partial(current.gpus_leftover_after, epsilon) - _largest_partial(best.gpus_leftover_after, epsilon)
            if abs(diff) > epsilon:
                return diff > 0
        elif rule in {"keep_variance_low", "keep_crumbiness_low"}:
            diff = _crumbiness(current.gpus_leftover_after, epsilon) - _crumbiness(best.gpus_leftover_after, epsilon)
            if abs(diff) > epsilon:
                return diff < 0

    return _lex_key(current.gpu_indices) < _lex_key(best.gpu_indices)


def _count_whole_gpus(gpus: Sequence[float], epsilon: float) -> int:
    """Count how many GPUs are whole (>= 1.0 - epsilon)."""
    return sum(1 for gpu in gpus if gpu >= 1.0 - epsilon)


def _largest_partial(gpus: Sequence[float], epsilon: float) -> float:
    """Find the largest partial GPU (< 1.0)."""
    partials = [gpu for gpu in gpus if gpu + epsilon < 1.0]
    return max(partials, default=0.0)


def _crumbiness(gpus: Sequence[float], epsilon: float) -> float:
    """Measure fragmentation 'crumbiness' (ratio of partial GPUs to total)."""
    total = sum(gpus)
    if total <= epsilon:
        return 0.0
    partial_sum = sum(gpu for gpu in gpus if gpu + epsilon < 1.0)
    return partial_sum / total


def _lex_key(indices: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Lexicographic key for final tie-breaker."""
    if not indices:
        return tuple()
    return tuple(indices)


__all__ = [
    "ClusterState",
    "Config",
    "Job",
    "Model",
    "PlacementDecision",
    "compute_fragmentation",
    "fgd_decide_placement",
    "commit_placement",
    "schedule",
]


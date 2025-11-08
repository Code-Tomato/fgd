"""Core Fragmentation Gradient Descent (FGD) algorithm.

This module contains only the core FGD algorithm from the paper:
- Fragmentation computation (Fn(M))
- Gradient descent placement decision

Reference: Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads 
with Fragmentation Gradient Descent (FGD)" (USENIX ATC '23)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Model:
    """Represents a model type with popularity."""
    model_name: str
    popularity: float


@dataclass
class Job:
    """Represents a job to be scheduled."""
    model_name: str
    gpu_demand: float


@dataclass
class ClusterState:
    """Represents the current state of GPU resources in the cluster."""
    gpus: List[float]

    def copy(self) -> "ClusterState":
        return ClusterState(self.gpus.copy())


@dataclass
class Config:
    """Minimal configuration needed for core FGD algorithm."""
    epsilon: float = 1e-9
    epsilon_int: float = 1e-6


# ---------------------------------------------------------------------------
# Core FGD Algorithm
# ---------------------------------------------------------------------------


def compute_fragmentation(
    cluster: ClusterState, workload: Sequence[Model], jobs: Sequence[Job], config: Config
) -> float:
    """Compute fragmentation rate f_cluster for the cluster.
    
    This implements Equation (1) from the paper: Fn(M) = Σ (pm / Σpm) * Fn(m)
    Returns the fragmentation rate (f_cluster), which is Fn(M) / U.
    
    Args:
        cluster: Current cluster state with GPU resources
        workload: Sequence of models with popularity weights
        jobs: Sequence of jobs to calculate average GPU demand per model
        config: Configuration with epsilon values
        
    Returns:
        Fragmentation rate (0.0 to 1.0)
    """
    gpus = cluster.gpus
    total_leftover = sum(gpus)
    total_popularity = sum(model.popularity for model in workload)
    if total_popularity <= config.epsilon:
        total_popularity = 1.0

    # Calculate average GPU demand per model from jobs
    model_demands: dict[str, list[float]] = {}
    for job in jobs:
        if job.model_name not in model_demands:
            model_demands[job.model_name] = []
        model_demands[job.model_name].append(job.gpu_demand)
    
    # Compute average demand per model
    avg_demands: dict[str, float] = {}
    for model_name, demands in model_demands.items():
        avg_demands[model_name] = sum(demands) / len(demands) if demands else 0.0

    expected_fragmentation = 0.0
    for model in workload:
        weight = model.popularity / total_popularity
        # Use average demand for this model, or 0.0 if no jobs exist
        avg_demand = avg_demands.get(model.model_name, 0.0)
        expected_fragmentation += weight * _fragmentation_amount_for_task(
            gpus, avg_demand, config
        )

    frag_rate = 0.0 if total_leftover <= config.epsilon else expected_fragmentation / total_leftover
    return frag_rate


def fgd_decide_placement(
    job: Job, cluster: ClusterState, workload: Sequence[Model], jobs: Sequence[Job], config: Config
) -> Optional[Tuple[int, ...]]:
    """Run Fragmentation Gradient Descent to decide job placement.
    
    This implements Algorithm 1 from the paper. For each possible placement,
    computes the fragmentation delta and selects the placement with the smallest
    increase in fragmentation.
    
    Args:
        job: Job to be placed
        cluster: Current cluster state
        workload: Sequence of models
        jobs: Sequence of all jobs (for fragmentation calculation)
        config: Configuration
        
    Returns:
        Tuple of GPU indices where the job should be placed, or None if infeasible
    """
    gpus_before = cluster.gpus.copy()
    frag_before = compute_fragmentation(cluster, workload, jobs, config)
    candidates = list(_enumerate_candidates(job, gpus_before, config))

    if not candidates:
        return None

    best_indices = None
    min_delta = float('inf')

    for candidate in candidates:
        simulated = ClusterState(gpus_before.copy())
        _apply_allocation(simulated.gpus, job, candidate, config)
        frag_after = compute_fragmentation(simulated, workload, jobs, config)
        delta = frag_after - frag_before

        if delta < min_delta:
            min_delta = delta
            best_indices = candidate

    return best_indices


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _classify_gpu_demand(value: float, config: Config) -> Tuple[str, float]:
    """Identify whether a demand is zero, fractional, or integer-sized."""
    if value < -config.epsilon:
        raise ValueError(f"GPU demand must be non-negative, got {value}")

    if abs(value) <= config.epsilon:
        return ("zero", 0.0)

    if value < 1.0 - config.epsilon:
        return ("fractional", value)

    k = int(round(value))
    if k <= 0:
        raise ValueError(f"Integer GPU demand must be >= 1, got {value}")

    if abs(value - k) <= config.epsilon_int:
        return ("integer", float(k))

    raise ValueError(
        "GPU demand must be fractional <= 1 or near an integer; "
        f"received {value}"
    )


def _fragmentation_amount_for_task(
    gpus: Sequence[float], gpu_demand: float, config: Config
) -> float:
    """Amount of leftover GPU that is useless to a given task class.
    
    This implements Equations (2)-(3) from the paper.
    """
    total_leftover = sum(gpus)
    if total_leftover <= config.epsilon:
        return 0.0

    mode, payload = _classify_gpu_demand(gpu_demand, config)

    if mode == "zero":
        return total_leftover

    if mode == "integer":
        k = int(payload)
        whole_slots = sum(1 for gpu in gpus if gpu >= 1.0 - config.epsilon)
        if whole_slots < k:
            return total_leftover
        return sum(gpu for gpu in gpus if gpu < 1.0 - config.epsilon)

    fractional = payload
    feasible = any(gpu >= fractional - config.epsilon for gpu in gpus)
    if not feasible:
        return total_leftover

    return sum(gpu for gpu in gpus if gpu + config.epsilon < fractional)


def _enumerate_candidates(
    job: Job, gpus: Sequence[float], config: Config
) -> Iterable[Optional[Tuple[int, ...]]]:
    """Generate all valid GPU placement candidates for a job."""
    epsilon = config.epsilon
    mode, payload = _classify_gpu_demand(job.gpu_demand, config)

    if mode == "zero":
        for idx in range(len(gpus)):
            yield (idx,)
        yield tuple()
        return

    if mode == "fractional":
        demand = payload
        for idx, leftover in enumerate(gpus):
            if leftover + epsilon >= demand:
                yield (idx,)
        return

    k = int(payload)
    whole_indices = [idx for idx, leftover in enumerate(gpus) if leftover + epsilon >= 1.0]
    if len(whole_indices) >= k:
        yield tuple(whole_indices[:k])


def _apply_allocation(
    gpus: List[float], job: Job, indices: Optional[Tuple[int, ...]], config: Config
) -> None:
    """Simulate allocating a job to specific GPU indices (modifies gpus in place)."""
    epsilon = config.epsilon
    mode, payload = _classify_gpu_demand(job.gpu_demand, config)

    if mode == "zero" or not indices:
        return

    if mode == "fractional":
        idx = indices[0]
        gpus[idx] = max(0.0, min(1.0, gpus[idx] - payload))
        return

    k = int(payload)
    for idx in indices[:k]:
        gpus[idx] = max(0.0, min(1.0, gpus[idx] - 1.0))


__all__ = [
    "ClusterState",
    "Config",
    "Job",
    "Model",
    "compute_fragmentation",
    "fgd_decide_placement",
]


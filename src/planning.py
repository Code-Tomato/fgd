"""Dynamic GPU planning for FGD scheduler.

This module provides functions to automatically determine the minimum number
of GPUs needed to schedule all jobs.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from .scheduler import Config, ClusterState, Job, Model, schedule


def plan_with_dynamic_g(
    jobs: Sequence[Job], workload: Sequence[Model], config: Config
) -> Tuple[int, Dict]:
    """Grow the number of GPUs until all jobs fit.
    
    Tries scheduling with increasing numbers of GPUs (1, 2, 3, ...) until
    all jobs can be scheduled. Returns the minimum number of GPUs needed.
    
    Args:
        jobs: Sequence of jobs to schedule
        workload: Sequence of models
        config: Configuration with g_max
        
    Returns:
        Tuple of (number_of_gpus, schedule_results_dict) with minimum GPUs needed
        
    Raises:
        RuntimeError: If unable to schedule all jobs within g_max GPUs
    """
    # Find the minimum number of GPUs where all jobs can fit
    for g in range(1, config.g_max + 1):
        cluster = ClusterState([1.0] * g)
        try:
            result = schedule(jobs, cluster, workload, config)
            # Found a solution where all jobs fit - return it
            return g, result
        except ValueError:
            continue
    
    # If we get here, couldn't fit all jobs
    raise RuntimeError(f"Unable to schedule all jobs within {config.g_max} GPUs")


__all__ = [
    "plan_with_dynamic_g",
]


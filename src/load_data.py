"""Load models, jobs, and cluster configuration from config/data files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .fgd import ClusterState, Job, Model

# Get project root (parent of src/)
_PROJECT_ROOT = Path(__file__).parent.parent

# Load jobs from CSV file.
def load_jobs(jobs_file: str | Path) -> list[Job]:
    jobs_path = Path(jobs_file)
    if not jobs_path.is_absolute():
        jobs_path = _PROJECT_ROOT / jobs_path
    
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")
    
    jobs = []
    with jobs_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            # Remove trailing comma if present, then split
            line = line.rstrip(",")
            parts = [p.strip() for p in line.split(",") if p.strip()]
            
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid format at line {line_num}: expected 'model_name,gpu_demand', "
                    f"got '{line}'"
                )
            
            model_name = parts[0]
            if not model_name:
                raise ValueError(
                    f"Invalid format at line {line_num}: model_name cannot be empty"
                )
            
            try:
                gpu_demand = float(parts[1])
            except ValueError:
                raise ValueError(
                    f"Invalid format at line {line_num}: gpu_demand must be a number, "
                    f"got '{parts[1]}'"
                )
            
            jobs.append(Job(model_name=model_name, gpu_demand=gpu_demand))
    
    return jobs

# Infer models from jobs and calculate popularity based on job distribution.
def get_models_from_jobs(jobs: Sequence[Job]) -> list[Model]:
    if not jobs:
        return []
    
    job_counts = Counter(job.model_name for job in jobs)
    total_jobs = len(jobs)
    
    return [
        Model(model_name=model_name, popularity=count / total_jobs)
        for model_name, count in sorted(job_counts.items())
    ]

# Load cluster configuration from device-config.json
def load_cluster(device_config_path: str | Path | None = None) -> ClusterState:
    
    if device_config_path is None:
        device_config_path = _PROJECT_ROOT / "config" / "device-config.json"
    else:
        device_config_path = Path(device_config_path)
        if not device_config_path.is_absolute():
            device_config_path = _PROJECT_ROOT / device_config_path
    
    with device_config_path.open("r") as f:
        config = json.load(f)
    
    # Extract number of GPUs
    num_gpus = config.get("num_gpus", 1)
    
    # Initialize cluster with full GPUs
    gpus = [1.0] * num_gpus
    return ClusterState(gpus=gpus)

__all__ = [
    "load_jobs",
    "load_cluster",
    "get_models_from_jobs",
]


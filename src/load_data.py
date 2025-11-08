"""Load models, jobs, and cluster configuration from config/data files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .fgd import ClusterState, Job, Model

# Get project root (parent of src/)
_PROJECT_ROOT = Path(__file__).parent.parent


def _get_model_id_to_name_map(models_config_path: str | Path | None = None) -> dict[int, str]:
    """Get mapping from model_id (1-indexed) to model_name from models-config.json.
    
    Args:
        models_config_path: Path to models-config.json file (defaults to config/models-config.json)
        
    Returns:
        Dictionary mapping model_id (int) to model_name (str)
    """
    if models_config_path is None:
        models_config_path = _PROJECT_ROOT / "config" / "models-config.json"
    else:
        models_config_path = Path(models_config_path)
        if not models_config_path.is_absolute():
            models_config_path = _PROJECT_ROOT / models_config_path
    
    with models_config_path.open("r") as f:
        config = json.load(f)
    
    model_map = {}
    for idx, model_name in enumerate(config["models"], start=1):
        model_map[idx] = model_name
    
    return model_map


def load_models(models_config_path: str | Path | None = None) -> list[Model]:
    """Load models from models-config.json.
    
    Extracts model names. Popularity is calculated later from job distribution.
    
    Args:
        models_config_path: Path to models-config.json file (defaults to config/models-config.json)
        
    Returns:
        List of Model objects with model_name and default popularity
    """
    if models_config_path is None:
        models_config_path = _PROJECT_ROOT / "config" / "models-config.json"
    else:
        models_config_path = Path(models_config_path)
        if not models_config_path.is_absolute():
            models_config_path = _PROJECT_ROOT / models_config_path
    
    with models_config_path.open("r") as f:
        config = json.load(f)
    
    models = []
    for model_name in config["models"]:
        # Default popularity to 1.0 (will be normalized later if needed)
        models.append(Model(model_name=model_name, popularity=1.0))
    
    return models


def load_jobs(
    jobs_file: str | Path,
    models_config_path: str | Path | None = None,
) -> list[Job]:
    """Load jobs from CSV file in Usher-style format.
    
    Format:
        First line: number of jobs
        Subsequent lines: model_id, gpu_demand
        Example:
            5
            1,0.25,
            2,0.5,
            3,0.75,
            1,0.25,
    
    Args:
        jobs_file: Path to jobs CSV file (required)
        models_config_path: Path to models-config.json for model_id mapping (defaults to config/models-config.json)
        
    Returns:
        List of Job objects
        
    Raises:
        FileNotFoundError: If jobs_file doesn't exist
        ValueError: If format is invalid or model_id not found
    """
    jobs_path = Path(jobs_file)
    if not jobs_path.is_absolute():
        jobs_path = _PROJECT_ROOT / jobs_path
    
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")
    
    with jobs_path.open("r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        return []
    
    # First line must be the number of jobs
    try:
        num_jobs = int(lines[0])
    except ValueError:
        raise ValueError(
            f"Invalid format: first line must be number of jobs, got '{lines[0]}'"
        )
    
    # Get model_id to model_name mapping
    model_map = _get_model_id_to_name_map(models_config_path)
    
    jobs = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Remove trailing comma if present
        if line.endswith(","):
            line = line[:-1]
        
        parts = [p.strip() for p in line.split(",") if p.strip()]
        
        if len(parts) < 2:
            continue
        
        # Format: model_id, gpu_demand
        model_id = int(parts[0])
        gpu_demand = float(parts[1])
        
        if model_id not in model_map:
            raise ValueError(
                f"Model ID {model_id} not found in models-config.json. "
                f"Available IDs: {sorted(model_map.keys())}"
            )
        model_name = model_map[model_id]
        
        jobs.append(Job(model_name=model_name, gpu_demand=gpu_demand))
    
    return jobs


def calculate_model_popularity(models: Sequence[Model], jobs: Sequence[Job]) -> list[Model]:
    """Calculate popularity for models based on job distribution.
    
    Args:
        models: List of models (with default popularity)
        jobs: List of jobs
        
    Returns:
        Updated list of models with popularity calculated from job distribution
    """
    if not jobs:
        # If no jobs, return models with equal popularity
        return list(models)
    
    # Count jobs per model
    job_counts = Counter(job.model_name for job in jobs)
    total_jobs = len(jobs)
    
    # Update popularity based on job distribution
    updated_models = []
    for model in models:
        count = job_counts.get(model.model_name, 0)
        popularity = count / total_jobs if total_jobs > 0 else 0.0
        updated_models.append(Model(model_name=model.model_name, popularity=popularity))
    
    return updated_models


def load_cluster(device_config_path: str | Path | None = None) -> ClusterState:
    """Load cluster configuration from device-config.json.
    
    Reads the number of GPUs from the config file.
    
    Args:
        device_config_path: Path to device-config.json file (defaults to config/device-config.json)
        
    Returns:
        ClusterState with GPUs initialized
    """
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
    "load_models",
    "load_jobs",
    "load_cluster",
    "calculate_model_popularity",
]


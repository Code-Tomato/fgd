"""Output generation for FGD scheduler.

This module handles formatting and writing GPU assignment results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .scheduler import ClusterState


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_indices(indices: Optional[Tuple[int, ...]]):
    """Format GPU indices for output (single index as int, multiple as list)."""
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    return list(indices)


# ---------------------------------------------------------------------------
# GPU Assignments Format (Usher-style)
# ---------------------------------------------------------------------------


def format_gpu_assignments(
    placements: Sequence[Dict], cluster: ClusterState, num_gpus: int
) -> str:
    """Format placements as GPU assignments in Usher-style format.
    
    Groups placements by GPU and shows which models are on each GPU.
    Includes fragmentation percentage for each GPU.
    
    Args:
        placements: List of placement dictionaries from schedule()
        cluster: Final cluster state
        num_gpus: Number of GPUs
        
    Returns:
        Formatted string showing GPU assignments
    """
    # Group placements by GPU
    gpu_assignments: Dict[int, List[Dict]] = {}
    
    for placement in placements:
        gpu_idx = placement["chosen_gpu_index"]
        if gpu_idx is None:
            continue
        
        # Handle both single GPU (int) and multiple GPUs (list)
        if isinstance(gpu_idx, int):
            gpu_indices = [gpu_idx]
        elif isinstance(gpu_idx, list):
            gpu_indices = gpu_idx
        else:
            continue
        
        # For jobs that span multiple GPUs, add to each GPU
        for idx in gpu_indices:
            if idx not in gpu_assignments:
                gpu_assignments[idx] = []
            gpu_assignments[idx].append(placement)
    
    # Build output string
    lines = []
    lines.append("=" * 60)
    lines.append("FINAL SCHEDULE:")
    lines.append("=" * 60)
    lines.append(f"\nTotal GPUs used: {num_gpus}\n")
    
    for gpu_id in range(num_gpus):
        gpu_leftover = cluster.gpus[gpu_id] if gpu_id < len(cluster.gpus) else 1.0
        gpu_demand_used = 1.0 - gpu_leftover
        gpu_demand_total = 1.0
        fragmentation_pct = gpu_leftover * 100.0
        
        lines.append(f"\nGPU(id={gpu_id}, gpu_demand={gpu_demand_used:.2f}/{gpu_demand_total:.2f}, fragmentation={fragmentation_pct:.2f}%)")
        
        if gpu_id in gpu_assignments:
            for placement in gpu_assignments[gpu_id]:
                model_name = placement["model_name"]
                gpu_demand = placement["gpu_demand"]
                lines.append(f"    - {model_name} (gpu_demand={gpu_demand})")
        else:
            lines.append("    (empty)")
    
    return "\n".join(lines)


def write_gpu_assignments(
    placements: Sequence[Dict],
    cluster: ClusterState,
    num_gpus: int,
    output_path: Path,
) -> Path:
    """Write GPU assignments to a readable text file.
    
    Args:
        placements: List of placement dictionaries from schedule()
        cluster: Final cluster state
        num_gpus: Number of GPUs
        output_path: Path to write output file
        
    Returns:
        Path to written file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = format_gpu_assignments(placements, cluster, num_gpus)
    
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(formatted)
        fh.write("\n")
    
    return output_path


__all__ = [
    "format_gpu_assignments",
    "write_gpu_assignments",
]


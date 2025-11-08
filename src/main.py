"""Example driver for the concise GPU-only FGD scheduler."""

from __future__ import annotations

import argparse
from pathlib import Path

from .load_data import (
    calculate_model_popularity,
    load_cluster,
    load_jobs,
    load_models,
)
from .fgd import ClusterState
from .scheduler import Config, schedule
from .output import write_gpu_assignments
from .planning import plan_with_dynamic_g


def process_single_file(
    input_file: Path,
    output_dir: Path,
    device_config_path: Path | None = None,
    models_config_path: Path | None = None,
    use_dynamic_gpus: bool = True,
) -> None:
    """Process a single input file and write outputs.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to write output files
        device_config_path: Path to device config (defaults to config/device-config.json)
        models_config_path: Path to models config (defaults to config/models-config.json)
        use_dynamic_gpus: If True, automatically scale up GPUs if jobs don't fit
    """
    # Load models and cluster
    models = load_models(models_config_path)
    cluster = load_cluster(device_config_path)
    
    # Load jobs from input file
    jobs = load_jobs(input_file, models_config_path)
    
    if not jobs:
        print(f"Warning: No jobs found in {input_file}")
        return
    
    # Calculate model popularity from job distribution
    models = calculate_model_popularity(models, jobs)
    
    config = Config()
    
    # Try to run scheduler, use dynamic GPUs if it fails
    try:
        results = schedule(jobs, cluster, models, config)
        num_gpus = len(cluster.gpus)
    except ValueError as e:
        if use_dynamic_gpus:
            # Try dynamic GPU planning
            print(f"Jobs don't fit in {len(cluster.gpus)} GPUs, trying dynamic GPU planning...")
            try:
                num_gpus, results = plan_with_dynamic_g(jobs, models, config)
                # Create cluster state for output
                cluster_final = ClusterState([1.0] * num_gpus)
                results = schedule(jobs, cluster_final, models, config)
                cluster = cluster_final
                print(f"Successfully scheduled with {num_gpus} GPUs")
            except RuntimeError as e2:
                print(f"Error: Unable to schedule jobs even with dynamic GPU planning: {e2}")
                raise
        else:
            print(f"Error: {e}")
            raise
    
    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write GPU assignments (readable format)
    output_name = input_file.stem
    assignments_path = output_dir / f"{output_name}.out.txt"
    write_gpu_assignments(
        results["placements"],
        cluster,
        num_gpus,
        assignments_path,
    )
    print(f"GPU assignments written to {assignments_path}")


def process_input_folder(
    input_folder: Path,
    output_dir: Path,
    device_config_path: Path | None = None,
    models_config_path: Path | None = None,
    use_dynamic_gpus: bool = True,
) -> None:
    """Process all CSV files in a folder.
    
    Args:
        input_folder: Folder containing input CSV files
        output_dir: Directory to write output files
        device_config_path: Path to device config (defaults to config/device-config.json)
        models_config_path: Path to models config (defaults to config/models-config.json)
        use_dynamic_gpus: If True, automatically scale up GPUs if jobs don't fit
    """
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    csv_files = sorted(input_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {input_folder}")
        return
    
    print(f"Processing {len(csv_files)} files from {input_folder}")
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        try:
            process_single_file(
                csv_file,
                output_dir,
                device_config_path,
                models_config_path,
                use_dynamic_gpus,
            )
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="FGD Scheduler")
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file or folder containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--no-dynamic-gpus",
        action="store_true",
        help="Don't automatically scale up GPUs if jobs don't fit",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            # Process single file
            process_single_file(
                input_path,
                output_dir,
                use_dynamic_gpus=not args.no_dynamic_gpus,
            )
        elif input_path.is_dir():
            # Process folder
            process_input_folder(
                input_path,
                output_dir,
                use_dynamic_gpus=not args.no_dynamic_gpus,
            )
        else:
            print(f"Error: {args.input} is not a valid file or directory")
    else:
        # Default: process input/ folder
        from .load_data import _PROJECT_ROOT
        default_inputs_folder = _PROJECT_ROOT / "input"
        
        if default_inputs_folder.exists() and default_inputs_folder.is_dir():
            # Process inputs folder by default
            process_input_folder(
                default_inputs_folder,
                output_dir,
                use_dynamic_gpus=not args.no_dynamic_gpus,
            )
        else:
            print("No input file specified and input/ folder not found")
            print("Usage: python3 -m src.main [--input <file_or_folder>] [--output <dir>]")
            print("Default: processes input/ folder")


if __name__ == "__main__":
    main()


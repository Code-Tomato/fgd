# FGD Scheduler

Fragmentation Gradient Descent (FGD) scheduler for GPU-sharing workloads.

**Reference**: Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent (FGD)" (USENIX ATC '23)

## Assumptions

- All GPUs are homogeneous (identical capacity)
- All GPUs start empty (no pre-existing allocations)
- CPU resources are not considered (GPU-only scheduling)

## Usage

```bash
# Process all files in input/ (default)
python3 -m src.main

# Process a specific file
python3 -m src.main --input input/test1.csv --output output/

# Process a folder
python3 -m src.main --input input/ --output output/
```

## Input Format

CSV files with one job per line:
```
resnet50,0.25,
mobilenet_v3_large,0.5,
vgg11,0.75,
resnet50,0.25,
mobilenet_v3_large,0.3,
```

Each line: `model_name,gpu_demand,`  

## Configuration

- `config/device-config.json`: Number of GPUs

## Output

GPU assignments showing which models are placed on each GPU, including fragmentation percentage per GPU.

# FGD Scheduler

Fragmentation Gradient Descent (FGD) scheduler for GPU-sharing workloads.

**Reference**: Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent (FGD)" (USENIX ATC '23)

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

CSV files in Usher-style format:
```
5
1,0.25,
2,0.5,
3,0.75,
1,0.25,
2,0.3,
```

First line: number of jobs  
Subsequent lines: `model_id,gpu_demand,`

## Configuration

- `config/device-config.json`: Number of GPUs
- `config/models-config.json`: List of model names

## Output

GPU assignments showing which models are placed on each GPU, including fragmentation percentage per GPU.

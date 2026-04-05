# Split Learning & Federated Learning Comparison

A research project benchmarking Split Learning (SL) and Split Federated Learning (SFL) 
algorithms across multiple image classification datasets. The experiments compare 
standard approaches against decoupled variants, tracking accuracy, 
communication cost, compute time, and GPU memory.

## Algorithms

| Script | Algorithm | Description |
|---|---|---|
| `sl_dsl.py` | **Standard SL** vs **DGL** | Standard Split Learning (single server-side backward pass) vs Decoupled Greedy Learning (each split trained independently with auxiliary classifiers) |
| `v1_sfl_dsfl.py` | **SFLv1** vs **DSFL** | SplitFed Learning V1 (FedAvg on client-side weights) vs Decoupled SplitFed Learning |
| `v2_sfl_dsfl.py` | **SFLv2** vs **DSFLv2** | SplitFed Learning V2 vs Decoupled SplitFed Learning V2 |

## Datasets

- **CIFAR-10** — 10-class image classification (32×32)
- **CIFAR-100** — 100-class image classification (32×32)
- **Tiny-ImageNet-200** — 200-class image classification (64×64)

Tiny-ImageNet-200 must be downloaded manually and placed at `data/tiny-imagenet-200/`.

## Model

All experiments use a **ResNet-110** backbone split into 2 partitions. Each partition is assigned an auxiliary classifier head used for local loss computation. The final classifier head is shared across all splits.

## Data Partitioning

Training data is distributed across clients using a **Dirichlet(α=0.5)** non-IID partition for SFL experiments, producing heterogeneous label distributions across clients. Standard SL experiments use a stratified IID split.

Client counts evaluated: **1, 5, 10** clients.

## Project Structure

```
sl-fl/
├── cifar-10/
│   ├── sl/
│   │   └── sl_dsl.py          # Standard SL vs DGL on CIFAR-10
│   └── sfl/
│       ├── v1_sfl_dsfl.py     # SFLv1 vs DSFL on CIFAR-10
│       └── v2_sfl_dsfl.py     # SFLv2 vs DSFLv2 on CIFAR-10
├── cifar-100/
│   ├── sl/
│   │   └── sl_dsl.py
│   └── sfl/
│       ├── v1_sfl_dsfl.py
│       └── v2_sfl_dsfl.py
├── tiny-200/
│   ├── sl/
│   │   └── sl_dsl.py
│   └── sfl/
│       ├── v1_sfl_dsfl.py
│       └── v2_sfl_dsfl.py
└── run.sh                     # SLURM job script to run all experiments
```

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
```

## Running Experiments

**Single script:**
```bash
python cifar-10/sl/sl_dsl.py
```

**All experiments (SLURM):**
```bash
sbatch run.sh
```

The SLURM script requests 1 GPU, 8 CPUs, and 32 GB RAM for up to 7 days.

## Outputs

Each script writes results to a `logs/` directory relative to where it is run:

- `logs/experiment_results.csv` — per-epoch metrics for all scenarios
- `logs/final_summary_table.csv` — aggregated summary statistics
- `logs/plots/` — PNG visualizations including:
  - Test/train accuracy convergence
  - Communication volume per epoch
  - Compute time breakdown (client fwd/bwd, server fwd/bwd)
  - Peak GPU memory usage
  - Epoch time distribution

## Metrics Tracked

| Metric | Description |
|---|---|
| `TrainAcc` / `TestAcc` | Training and test accuracy (%) |
| `CommFwdBytes` / `CommBwdBytes` | Bytes transferred at the split layer (forward smashed data / backward gradients) |
| `ClientTimeFwd` / `ClientTimeBwd` | Client-side compute time for forward and backward pass |
| `ServerTimeFwd` / `ServerTimeBwd` | Server-side compute time for forward and backward pass |
| `CommTimeWall` | Wall-clock time not attributable to compute (communication overhead) |
| `TotalTime` | Total epoch wall-clock time |
| `PeakGPUMemMB` | Peak GPU memory usage in MB |
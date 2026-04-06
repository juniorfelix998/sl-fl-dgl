#!/bin/bash
#SBATCH --job-name=sl-fl-all
#SBATCH --partition=gpu-week-long
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err

# Activate conda
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate venv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    echo "=== Running $1 ==="
    srun --exclusive -n1 python -u "$SCRIPT_DIR/$1"
    echo "=== Done: $1 ==="
}

# CIFAR-10
run cifar-10/sl/split-2/sl_dsl.py
run cifar-10/sl/split-3/sl_dsl.py
run cifar-10/sl/split-4/sl_dsl.py
run cifar-10/sfl/split-2/v1/v1_sfl_dsfl.py
run cifar-10/sfl/split-2/v2/v2_sfl_dsfl.py
run cifar-10/sfl/split-3/v1/v1_sfl_dsfl.py
run cifar-10/sfl/split-3/v2/v2_sfl_dsfl.py
run cifar-10/sfl/split-4/v1/v1_sfl_dsfl.py
run cifar-10/sfl/split-4/v2/v2_sfl_dsfl.py

# CIFAR-100
run cifar-100/sl/split-2/sl_dsl.py
run cifar-100/sl/split-3/sl_dsl.py
run cifar-100/sl/split-4/sl_dsl.py
run cifar-100/sfl/split-2/v1/v1_sfl_dsfl.py
run cifar-100/sfl/split-2/v2/v2_sfl_dsfl.py
run cifar-100/sfl/split-3/v1/v1_sfl_dsfl.py
run cifar-100/sfl/split-3/v2/v2_sfl_dsfl.py
run cifar-100/sfl/split-4/v1/v1_sfl_dsfl.py
run cifar-100/sfl/split-4/v2/v2_sfl_dsfl.py

# Tiny-ImageNet-200
run tiny-200/sl/split-2/sl_dsl.py
run tiny-200/sl/split-3/sl_dsl.py
run tiny-200/sl/split-4/sl_dsl.py
run tiny-200/sfl/split-2/v1/v1_sfl_dsfl.py
run tiny-200/sfl/split-2/v2/v2_sfl_dsfl.py
run tiny-200/sfl/split-3/v1/v1_sfl_dsfl.py
run tiny-200/sfl/split-3/v2/v2_sfl_dsfl.py
run tiny-200/sfl/split-4/v1/v1_sfl_dsfl.py
run tiny-200/sfl/split-4/v2/v2_sfl_dsfl.py

echo "All runs completed."

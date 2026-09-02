#!/bin/bash
#SBATCH --job-name=fjsp_test
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/test_seed%a_%j.out
#SBATCH --error=logs/test_seed%a_%j.err

# Evaluates one model_comparison_experiment model (seed = array task id)
# against every dataset under ./data/data_test, greedy + sampling.
# Submit with: sbatch run_test.sh   (after run_train.sh has finished)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

SEED=${SLURM_ARRAY_TASK_ID}
N_J=20
N_M=10
DATA_SOURCE=SD1

# eligible-operation pooling (keeps only eligible operations) -- must match run_train.sh
POOLING_TYPE=eligible
export RUN_TAG=eligible

MODEL_PATH="./trained_network/${DATA_SOURCE}/${N_J}x${N_M}+${RUN_TAG}_seed${SEED}.pth"

# Verify the model exists before launching
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: No model file found at ${MODEL_PATH}" >&2
    exit 1
fi

echo "=== Testing seed=${SEED} model=${MODEL_PATH} pooling=${POOLING_TYPE} ==="
srun nvidia-smi || true

exec srun "${PYTHON}" run_test_suite.py \
    --seeds "${SEED}" \
    --pooling_type "${POOLING_TYPE}" \
    --device cuda

#!/bin/bash
#SBATCH -J sense-generate
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out

# SENSE pilot — batch generation on Bridges-2.
# PRE-CONDITION: data shards already downloaded locally (compute nodes have
# limited/no egress — do NOT rely on streaming or HF downloads inside this job).

set -euo pipefail

module load anaconda3
module load cuda/12.4.0
conda activate $PROJECT/envs/terramind311

export HF_HOME=$PROJECT/.huggingface        # weights already cached here by smoke test
export HF_HUB_OFFLINE=1                     # fail fast if anything tries the network

cd $PROJECT/sense-pilot
python src/generate.py --config configs/pilot.yaml

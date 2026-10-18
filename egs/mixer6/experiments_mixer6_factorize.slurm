#!/usr/bin/env bash
# Build aligned Mixer 6 XS blocks and the rank-75 factorization.
# Submit directly, or use submit_mixer6_utxss.sh to chain the CPU UTXSS job.
#SBATCH --job-name=linearvc_mixer6_r75
#SBATCH --output=/home/hltcoe/xli/ARTS/linearvc/egs/mixer6/linearvc_mixer6_r75_%j.txt
#SBATCH --time=72:00:00
#SBATCH --mail-user=xli257@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100

set -euo pipefail
set +u
source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh
conda activate cftts_2.8
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd /home/hltcoe/xli/ARTS
python /home/hltcoe/xli/ARTS/linearvc/egs/mixer6/experiments_mixer6.py \
    --manifest /exp/xli/ARTS/ARTS_eval/exp/privacy_attributes/ldc_train_local_eval/preflight/ldc_train_manifest.jsonl \
    --anchor-xs /home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/librispeech_train-clean-100/rank_75/src_103/XS/103.npy \
    --anchor-speaker 103 \
    --rank 75 \
    --working-rank 1024 \
    --output-root /home/hltcoe/xli/ARTS/linearvc/exp/content_factorization \
    --corpus-name mixer6_ldc_train_local_eval \
    --device cuda

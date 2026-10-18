#!/usr/bin/env bash
# Factorize all 251 LibriSpeech and 45 Mixer 6 XS blocks.
#SBATCH --job-name=linearvc_mixer6_libri_r75
#SBATCH --output=/home/hltcoe/xli/ARTS/linearvc/egs/mixer6/linearvc_mixer6_libri_r75_%j.txt
#SBATCH --time=48:00:00
#SBATCH --mail-user=xli257@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=cpu

set -euo pipefail
set +u
source /home/hltcoe/xli/.bashrc
source /home/hltcoe/xli/anaconda3/etc/profile.d/conda.sh
conda activate cftts_2.8
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

combined_dir=/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/mixer6_librispeech_train-clean-100/rank_75/src_103
cd /home/hltcoe/xli/ARTS
python /home/hltcoe/xli/ARTS/linearvc/egs/mixer6/experiments_mixer6.py \
    --manifest "${combined_dir}/missing_librispeech_manifest.jsonl" \
    --anchor-xs /home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/librispeech_train-clean-100/rank_75/src_103/XS/103.npy \
    --anchor-speaker 103 \
    --rank 75 \
    --working-rank 1024 \
    --output-root /home/hltcoe/xli/ARTS/linearvc/exp/content_factorization \
    --corpus-name mixer6_librispeech_train-clean-100 \
    --factorize-speakers-file "${combined_dir}/all_speakers.txt" \
    --factorize-only

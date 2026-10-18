#!/usr/bin/env bash
# Compute UTXSS for the combined LibriSpeech + Mixer 6 factorization.
#SBATCH --job-name=linearvc_mixer6_libri_utxss
#SBATCH --output=/home/hltcoe/xli/ARTS/linearvc/egs/mixer6/linearvc_mixer6_libri_utxss_%j.txt
#SBATCH --time=24:00:00
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

cd /home/hltcoe/xli/ARTS
python -m linearvc.egs.librispeech.experiments_libri_inverse \
    --transform_root_dir /home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/mixer6_librispeech_train-clean-100/rank_75/src_103 \
    --transform_type UTXSS \
    --xs_folder \
    --streaming-qr

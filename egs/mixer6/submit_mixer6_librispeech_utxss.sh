#!/usr/bin/env bash
# Submit preparation/alignment, factorization, then UTXSS as dependent jobs.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prepare_job=$(sbatch --parsable "${script_dir}/experiments_mixer6_librispeech_prepare.slurm")
factor_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" "${script_dir}/experiments_mixer6_librispeech_factorize.slurm")
inverse_job=$(sbatch --parsable --dependency="afterok:${factor_job}" "${script_dir}/experiments_mixer6_librispeech_inverse.slurm")

printf 'Preparation/alignment job: %s\nFactorization job: %s\nUTXSS job: %s\n' \
    "${prepare_job}" "${factor_job}" "${inverse_job}"

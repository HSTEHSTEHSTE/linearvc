#!/usr/bin/env bash
# Submit the GPU factorization followed by the dependent CPU UTXSS solve.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
factor_job=$(sbatch --parsable "${script_dir}/experiments_mixer6_factorize.slurm")
inverse_job=$(sbatch --parsable --dependency="afterok:${factor_job}" "${script_dir}/experiments_mixer6_inverse.slurm")

printf 'Factorization job: %s\nUTXSS job: %s\n' "${factor_job}" "${inverse_job}"

#!/usr/bin/env bash
# Shared deterministic runtime baseline for LinearVC Slurm launchers.
# Source this after PROJECT_DIR is set and before activating Conda.

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PATH=/opt/mprov/cloack/slurm/current/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

JOB_USER="${SLURM_JOB_USER:-${USER:-}}"
if [[ -z "$JOB_USER" ]]; then
    echo "Could not determine the Slurm job user." >&2
    return 1
fi
export USER="$JOB_USER"

if [[ -z "${HOME:-}" || ! -d "${HOME:-}" ]]; then
    HOME="$(getent passwd "$JOB_USER" | awk -F: 'NR == 1 { print $6 }')"
    if [[ -z "$HOME" || ! -d "$HOME" ]]; then
        echo "Could not determine HOME for $JOB_USER." >&2
        return 1
    fi
    export HOME
fi

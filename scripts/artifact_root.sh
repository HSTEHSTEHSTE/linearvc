#!/usr/bin/env bash
# Shared location for runtime artifacts. Source code stays in the checkout;
# checkpoints, data, and caches belong in nandrew9 scratch. Slurm's combined
# session logs remain in the workspace from which each job was submitted.

: "${LINEARVC_ARTIFACT_ROOT:=/weka/scratch/jhu/nandrew9/xli257/linearvc}"
export LINEARVC_ARTIFACT_ROOT

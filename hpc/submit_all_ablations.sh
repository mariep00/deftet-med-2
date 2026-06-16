#!/bin/sh
# Submit all loss ablation variants as separate LSF jobs.
# Usage: bash hpc/submit_all_ablations.sh
# Override defaults with env vars, e.g.:
#   WANDB_MODE=online bash hpc/submit_all_ablations.sh

VARIANTS="
loss00_recon_only
loss01_plus_lap
loss02_plus_delta
loss03_plus_volume
loss04_plus_amips
loss05_plus_smooth
"

for VARIANT in $VARIANTS; do
    echo "==> Submitting $VARIANT"
    ABLATION_VARIANT="$VARIANT" bsub < hpc/submit_loss_ablation.sh
done

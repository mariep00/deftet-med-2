#!/bin/sh
# ===== LSF options =====
#BSUB -J deftet_loss_ablation
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -M 7GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -W 24:00
##BSUB -u s233736@tu.dk
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

# ===== User config =====
DATASET_DIR="${DATASET_DIR:-/work3/s233736/datasets/mesh_surfaces_crop_full}"
SPLIT_DIR="${SPLIT_DIR:-$HOME/thesis/deftet-med/splits/surfaces_80_10_10}"
ABLATION_VARIANT="${ABLATION_VARIANT:-loss00_recon_only}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RES="${RES:-120}"
EPOCHS="${EPOCHS:-100}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-10}"
INPUT_POINTS="${INPUT_POINTS:-10000}"
PRINT_EVERY="${PRINT_EVERY:-10}"
WANDB_ENTITY="${WANDB_ENTITY:-s233736-danmarks-tekniske-universitet-dtu}"
WANDB_PROJECT="${WANDB_PROJECT:-deftet-med}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/work3/s233736/deftet_runs}"

# Supplement-style common reconstruction weights.
LAMBDA_DEF=1
LAMBDA_OCC=1
LAMBDA_SURF=10
LAMBDA_SURF_CHAMFER=10
LAMBDA_EDGE=0
LAMBDA_LAP_V_LOSS=0
POW=4

# Sequential regularizer weights from the DefTet supplement.
LAMBDA_LAP=0
LAMBDA_DELTA=0
LAMBDA_AREA=0
LAMBDA_AMIPS=0
LAMBDA_NORMAL=0

case "$ABLATION_VARIANT" in
  loss00_recon_only)
    ;;
  loss01_plus_lap)
    LAMBDA_LAP=1e-3
    ;;
  loss02_plus_delta)
    LAMBDA_LAP=1e-3
    LAMBDA_DELTA=1e-3
    ;;
  loss03_plus_volume)
    LAMBDA_LAP=1e-3
    LAMBDA_DELTA=1e-3
    LAMBDA_AREA=1
    ;;
  loss04_plus_amips)
    LAMBDA_LAP=1e-3
    LAMBDA_DELTA=1e-3
    LAMBDA_AREA=1
    LAMBDA_AMIPS=1e-5
    ;;
  loss05_plus_smooth)
    LAMBDA_LAP=1e-3
    LAMBDA_DELTA=1e-3
    LAMBDA_AREA=1
    LAMBDA_AMIPS=1e-5
    LAMBDA_NORMAL=1e-2
    ;;
  *)
    echo "Unknown ABLATION_VARIANT: $ABLATION_VARIANT"
    echo "Expected one of:"
    echo "  loss00_recon_only"
    echo "  loss01_plus_lap"
    echo "  loss02_plus_delta"
    echo "  loss03_plus_volume"
    echo "  loss04_plus_amips"
    echo "  loss05_plus_smooth"
    exit 1
    ;;
esac

EXP_ID="${ABLATION_VARIANT}_res${RES}"

echo "==> Ablation variant: $ABLATION_VARIANT"
echo "==> Experiment ID: $EXP_ID"
echo "==> Loss weights:"
echo "    lambda_def=$LAMBDA_DEF"
echo "    lambda_occ=$LAMBDA_OCC"
echo "    lambda_surf=$LAMBDA_SURF"
echo "    lambda_surf_chamfer=$LAMBDA_SURF_CHAMFER"
echo "    lambda_lap=$LAMBDA_LAP"
echo "    lambda_delta=$LAMBDA_DELTA"
echo "    lambda_area=$LAMBDA_AREA"
echo "    lambda_amips=$LAMBDA_AMIPS"
echo "    lambda_normal=$LAMBDA_NORMAL"

# ===== Load CUDA =====
module load gcc/9.5.0-binutils-2.38
module load cuda/11.1

# ===== Env setup =====
source ~/miniforge3/bin/activate
conda activate vdeftet

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"
export OMP_NUM_THREADS="$LSB_DJOB_NUMPROC"

cd ~/thesis/deftet-med || { echo "Project dir not found"; exit 1; }

if [ "$WANDB_MODE" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Run 'wandb login' on the HPC or export WANDB_API_KEY before submitting."
  echo "For no-internet jobs, submit with: WANDB_MODE=offline bsub < hpc/submit_loss_ablation.sh"
  exit 1
fi

# ===== Run training =====
python train_multigpu.py \
  --pow "$POW" \
  --save_vis \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --val-every "$VAL_EVERY" \
  --input_points "$INPUT_POINTS" \
  --print_every "$PRINT_EVERY" \
  --loader_workers "$LOADER_WORKERS" \
  --dataset_dir "$DATASET_DIR" \
  --train_split_file "$SPLIT_DIR/train.txt" \
  --val_split_file "$SPLIT_DIR/val.txt" \
  --test_split_file "$SPLIT_DIR/test.txt" \
  --save_vis_every 5000 \
  --save_val_surfaces_last_n 10 \
  --no_use_pos_encoding \
  --no_use_vert_feat \
  --use_init_pos_mask \
  --point_cloud \
  --lambda_def "$LAMBDA_DEF" \
  --lambda_occ "$LAMBDA_OCC" \
  --lambda_surf "$LAMBDA_SURF" \
  --lambda_surf_chamfer "$LAMBDA_SURF_CHAMFER" \
  --lambda_edge "$LAMBDA_EDGE" \
  --lambda_lap "$LAMBDA_LAP" \
  --lambda_delta "$LAMBDA_DELTA" \
  --lambda_area "$LAMBDA_AREA" \
  --lambda_amips "$LAMBDA_AMIPS" \
  --lambda_normal "$LAMBDA_NORMAL" \
  --lambda_lap_v_loss "$LAMBDA_LAP_V_LOSS" \
  --res "$RES" \
  --no_expand_boundary \
  --use_two_encoder \
  --no_use_pvcnn_pos_decoder \
  --no_use_dvr_pos_decoder \
  --use_gcn_pos_decoder \
  --no_use_dvr_occ_decoder \
  --add_input_noise \
  --use_pvcnn_occ_decoder \
  --experiment_id "$EXP_ID" \
  --scale_pvcnn \
  --wandb \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$EXP_ID" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_log_every "$WANDB_LOG_EVERY" \
  --experiment_path "$EXPERIMENT_ROOT"

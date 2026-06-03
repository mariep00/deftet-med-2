#!/bin/sh
# ===== LSF options =====
#BSUB -J deftet_train
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
DATASET_DIR="/work3/s233736/datasets/mesh_surfaces_full"
SPLIT_DIR="$HOME/thesis/deftet-med/splits/surfaces_80_10_10"
EXP_ID="mri15-5"   # change if you want a new run name
BATCH_SIZE=1 # chnaged from 8 
RES=100
PRINT_EVERY=10

# ===== Load CUDA (MANDATORY) =====
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


# ===== Run training =====
python train_multigpu.py \
  --pow 4 \
  --save_vis \
  --batch_size "$BATCH_SIZE" \
  --print_every "$PRINT_EVERY" \
  --dataset_dir "$DATASET_DIR" \
  --train_split_file "$SPLIT_DIR/train.txt" \
  --val_split_file "$SPLIT_DIR/val.txt" \
  --test_split_file "$SPLIT_DIR/test.txt" \
  --save_vis_every 1000 \
  --save_val_surfaces_last_n 10 \
  --no_use_pos_encoding \
  --no_use_vert_feat \
  --use_init_pos_mask \
  --point_cloud \
  --lambda_surf 5 \
  --lambda_surf_chamfer 1 \
  --lambda_amips 1 \
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
  --experiment_path "/work3/s233736/deftet_runs" 

  #changed from --save_vis_every 10000

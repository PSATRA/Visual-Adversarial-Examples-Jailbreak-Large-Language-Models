#!/usr/bin/env bash
# ------------------------------------------------------------
# run_gemma3_visual_attack.sh
#
# This script calls gemma3_visual_attack.py and passes the necessary command-line arguments
# Usage:
#   bash run_gemma3_visual_attack.sh
# ------------------------------------------------------------

# —— Configuration Section —— 
# Replace the following variables with your actual paths:
PYTHON_BIN="/gpfs-flash/hulab/zhangwei_srt/miniconda3/envs/minigpt4/bin/python"
SCRIPT_PATH="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/gemma3_visual_attack.py"

# Path to the CSV data
INSTRUCT_PAIRS="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/harmful_corpus/JBB-harmful-behaviors.csv"

# Path to the clean input image
CLEAN_IMG="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/adversarial_images/clean.jpeg"

# Path to the Gemma3 checkpoint
MODEL_CKPT="/gpfs-flash/hulab/zhangwei_srt/lige/weights/gemma-3-4b-it"

# Directory to save the adversarial results
SAVE_DIR_ADV_IMG="/gpfs-flash/hulab/zhangwei_srt/lige/princeton/gemma3-4b-unconstrained"

# Device to use (GPU/CPU)
DEVICE="cuda:7"

# APGD hyperparameters
NUM_ITER=2000
ALPHA_INIT=0.5
EPSILON=1.0
RHO=0.75

# Random seed
SEED=17

# —— Script Invocation Section —— 
$PYTHON_BIN $SCRIPT_PATH \
  --seed $SEED \
  --instruct_pairs "$INSTRUCT_PAIRS" \
  --clean_img "$CLEAN_IMG" \
  --model_ckpt "$MODEL_CKPT" \
  --save_dir_adv_img "$SAVE_DIR_ADV_IMG" \
  --device "$DEVICE" \
  --num_iter $NUM_ITER \
  --alpha_init $ALPHA_INIT \
  --epsilon $EPSILON \
  --rho $RHO
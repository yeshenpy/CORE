#!/usr/bin/env bash
# CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
# Paper: https://openreview.net/forum?id=86IvZmY26S
# Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
#          Zhentao Tang, Mingxuan Yuan, Junchi Yan
# License: Non-Commercial License (see LICENSE). Commercial use requires permission.
# Signature: CORE Authors (NeurIPS 2025)

set -euo pipefail

mkdir -p logs

# Pure HPWL optimization (reward uses `--L_weight` in CORE.py)
# You can override these env vars:
# - DEVICE: GPU id, set -1 for CPU (default: 0)
# - SEED: random seed (default: 1)
# - RESULT_DIR: output root dir (default: ./results-hpwl)
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1}"
RESULT_DIR="${RESULT_DIR:-./results-hpwl}"

export WANDB_MODE="${WANDB_MODE:-offline}"

# n10  
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "n10"   --one_epsodic_length=10   --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=60 --EA_iter=6  > ./logs/hpwl_n10.log 2>&1 &
# n30  
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "n30"   --one_epsodic_length=30   --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=90 --EA_iter=27 > ./logs/hpwl_n30.log 2>&1 &
# n50  
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "n50"   --one_epsodic_length=50   --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=60 --EA_iter=30 > ./logs/hpwl_n50.log 2>&1 &
# n100 
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "n100"  --one_epsodic_length=100  --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=6  --EA_iter=6  > ./logs/hpwl_n100.log 2>&1 &
# n200
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "n200"  --one_epsodic_length=200  --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=6  --EA_iter=12 > ./logs/hpwl_n200.log 2>&1 &
# n300 
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "n300"  --one_epsodic_length=300  --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=6  --EA_iter=18 > ./logs/hpwl_n300.log 2>&1 &
# ami49 
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "ami49" --one_epsodic_length=49   --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=90 --EA_iter=45 > ./logs/hpwl_ami49.log 2>&1 &
# ami33 
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICE" nohup python CORE.py --circuit "ami33" --one_epsodic_length=33   --gap_iter_update_temperature 1 --device="$DEVICE" --seed="$SEED" --result_dir="$RESULT_DIR" --enable_draw=0 --weight_hpwl=1.0 --weight_area=0.0 --weight_feedthrough=0.0 --EA_Area_ratio=0.0 --A_weight=0.0 --O_weight=0.0 --L_weight=1.0 --ppo_pop_size=90 --EA_iter=30 > ./logs/hpwl_ami33.log 2>&1 &

echo "Launched CORE HPWL runs. Logs: ./logs/hpwl_*.log"
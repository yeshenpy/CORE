#!/bin/bash
# ================================================================================
# CORE: Collaborative Optimization with RL and EA for Floorplanning
# NeurIPS 2025 | https://github.com/yeshenpy/CORE
# ================================================================================

# Example: Train on n300 circuit with 15 parallel environments
python CORE.py --circuit="n300" \
    --one_epsodic_length=300 \
    --device=0 \
    --L_weight=1e-4 \
    --A_weight=0.0 \
    --O_weight=0.0 \
    --gamma=0.999 \
    --clip-vloss \
    --add_res \
    --num_envs=15 \
    --clip-coef=0.1 \
    --ent_coef=0.01 \
    --seed=1

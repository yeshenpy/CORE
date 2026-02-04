# CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://openreview.net/forum?id=86IvZmY26S)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-MCNC%20%2B%20GSRC-orange.svg)](#benchmarks)

**EA+RL hybrid framework** for B\*-Tree floorplanning, reproducing the approach described in the NeurIPS 2025 paper and reporting strong results on MCNC and GSRC benchmarks.

> **CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning**  
> Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang, Zhentao Tang, Mingxuan Yuan, Junchi Yan  
> *NeurIPS 2025*

To the best of our knowledge, this is the **first open-source RL-based floorplanning algorithm**, and also the **first open-source floorplanning environment with a C++ backend directly callable from Python** (via `pybind11`).

> **⭐ Why this repo matters**
>
> - **Open-source RL floorplanning** (to the best of our knowledge)
> - **Python-callable C++ backend environment** (B\*-Tree via `pybind11`)



## 🎯 Highlights

- 🥇 **Strong benchmark performance**: Reports **12.9% improvement in wirelength** on standard benchmarks (see paper)
- 🚀 **First ERL Algorithm for Floorplanning**: Pioneering evolutionary reinforcement learning approach
- ⚡ **Efficient Optimization**: Synergizes Evolutionary Algorithms (EAs) and Reinforcement Learning (RL)
- 📊 **Comprehensive Benchmarks**: Evaluated on MCNC and GSRC standard benchmarks

## Abstract

Floorplanning is the initial step in the physical design process of Electronic Design Automation (EDA), directly influencing subsequent placement, routing, and final power of the chip. However, the solution space in floorplanning is vast, and current algorithms often struggle to explore it sufficiently, making them prone to getting trapped in local optima. To achieve efficient floorplanning, we propose **CORE**, a general and effective solution optimization framework that synergizes Evolutionary Algorithms (EAs) and Reinforcement Learning (RL) for high-quality layout search and optimization. Specifically, we propose the Clustering-based Diversified Evolutionary Search that directly perturbs layouts and evolves them based on novelty and performance. Additionally, we model the floorplanning problem as a sequential decision problem with B\*-Tree representation and employ RL for efficient learning. To efficiently coordinate EAs and RL, we propose the reinforcement-driven mechanism and evolution-guided mechanism. The former accelerates population evolution through RL, while the latter guides RL learning through EAs. The experimental results on the MCNC and GSRC benchmarks demonstrate that CORE outperforms other strong baselines in terms of wirelength and area utilization metrics, achieving a **12.9% improvement in wirelength**.

## Key Features

- 🧬 **Clustering-based Diversified Evolutionary Search**: Directly perturbs layouts and evolves them based on novelty and performance
- 🌳 **B\*-Tree Representation**: Models floorplanning as a sequential decision problem
- 🔄 **Reinforcement-driven Mechanism**: Accelerates population evolution through RL
- 📈 **Evolution-guided Mechanism**: Guides RL learning through EAs
- 🏗️ **First ERL Algorithm for Floorplanning**: Surpasses existing RL-based methods

## Project Structure

```
CORE/
├── src/                    # C++ source code for B*-Tree floorplanner
│   ├── floorplanner.cpp/h  # Main floorplanner implementation
│   ├── block.cpp/h         # Block data structure
│   ├── node.cpp/h          # B*-Tree node
│   ├── net.cpp/h           # Netlist handling
│   └── pybind.cpp          # Python bindings
├── input_pa2/              # Benchmark circuits (MCNC & GSRC)
├── utils/                  # Utility functions
├── CORE.py                # Main entrypoint: EA+RL hybrid framework ⭐
├── SA.py                   # Simulated Annealing baseline
├── EA_with_PPO.py          # Experimental EA variant (not the main entrypoint)
├── PPO_utils.py            # PPO helpers (masks, data conversion, training)
├── Net.py                  # Neural network architectures (GNN + Transformer)
├── Makefile               # Build configuration
├── environment.yaml       # Conda environment specification
├── new_run.sh              # Example batch script for HPWL-focused runs
└── README.md              # This file
```

## Requirements

- Python 3.8
- PyTorch (CUDA recommended)
- PyTorch Geometric
- pybind11
- g++ (for compiling C++ code)

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate GPU_EDA
```

Note: `environment.yaml` is provided as a reference environment snapshot and may need adjustments on your machine (OS/CUDA/PyTorch/PyG versions). If the environment creation fails, please create a clean Python environment first, then install **PyTorch** and **PyTorch Geometric** with mutually compatible versions for your system.

### 2. Compile C++ Code

The B\*-Tree floorplanner is implemented in C++ with Python bindings:

```bash
make
```

This generates the `tree` Python module (Linux: `tree.cpython-*.so`, Windows: `tree*.pyd`) that provides the B\*-Tree floorplanner functionality.

> Note on platform support: the provided `Makefile` is written for a typical Linux toolchain (`g++`, `python3-config`). If you are on Windows, we recommend using WSL2 (Ubuntu) to build the `tree` module.

### 3. Verify Installation

```bash
python -c "import tree; print('CORE installed successfully!')"
```

## Usage

### Training with CORE (Recommended)



You can also launch a batch of HPWL-focused runs (MCNC + GSRC) with:

```bash
bash new_run.sh
```

This script uses `nohup` and backgrounds multiple runs, so it is intended for Linux servers/workstations.

### Training with Simulated Annealing (Baseline)

```bash
python SA.py --circuit="n300" \
             --weight_hpwl=1.0 \
             --weight_area=0.0 \
             --gap_iter_update_temperature=1000
```

### Key Arguments

For the full list, run `python CORE.py --help` (or `SA.py --help`).

| Argument | Description | Default    |
|----------|-------------|------------|
| `--circuit` | Circuit name (n10, n30, n50, n100, n200, n300, ami33, ami49, etc.) | n300       |
| `--one_epsodic_length` | Episode length (number of B\*-Tree insertion steps) | 300        |
| `--device` | GPU id (`-1` for CPU). Also controls `CUDA_VISIBLE_DEVICES`. | 0          |
| `--ppo_pop_size` | Number of PPO worker processes (parallel rollouts) | 6          |
| `--pop_size` | EA population size | 100        |
| `--num_cluster` | KMeans clusters for diversified evolution | 4          |
| `--num_best` | Elites kept per cluster | 2          |
| `--EA_iter` | EA inner iterations per epoch | 18         |
| `--total_epoch` | Total outer epochs | 10000      |
| `--L_weight` | Reward weight for HPWL term | defalut    |
| `--A_weight` | Reward weight for area ratio term | 0.0 or 5.0 |
| `--O_weight` | Reward weight for outbound penalty term | 0.0        |
| `--gamma` | PPO discount factor | 0.999      |
| `--batch_size` | PPO batch size | defalut    |
| `--ent_coef` | PPO entropy coefficient | defalut    |
| `--seed` | Random seed | 1          |
| `--clip-vloss` | Enable PPO value-loss clipping | clip-vloss      |

## Benchmarks

The `input_pa2/` directory contains benchmark circuits from MCNC and GSRC:

| Benchmark | Circuits |
|-----------|----------|
| **MCNC** | ami33, ami49, apte, hp, xerox |
| **GSRC** | n10, n30, n50, n100, n200, n300 |

Each circuit has two files:
- `*.block`: Block definitions (name, width, height)
- `*.nets`: Netlist connections

## Results

CORE achieves **state-of-the-art performance** on standard floorplanning benchmarks:

| Method | Wirelength Improvement |
|--------|----------------------|
| Previous best (reported in paper) | Baseline |
| **CORE (Ours)** | **-12.9%** ⬇️ |

> CORE represents the **first evolutionary reinforcement learning (ERL) algorithm for floorplanning**, surpassing all existing RL-based and traditional methods.

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{li2025core,
  title={CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning},
  author={Li, Pengyi and Kai, Shixiong and Hao, Jianye and Zhong, Ruizhe and Tang, Hongyao and Tang, Zhentao and Yuan, Mingxuan and Yan, Junchi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgments

The B\*-Tree implementation is adapted from [B-Star-Tree](https://github.com/Ashley990921/B-Star-Tree).

## License

This project is licensed under a **Non-Commercial License with Commercial Use Restriction**.

**For Non-Commercial Use:**
- ✅ You are free to use, copy, modify, merge, publish, and distribute this work
- ✅ You are free to prepare derivative works
- ✅ Subject to attribution and license inclusion requirements

**For Commercial Use:**
- ⚠️ **Commercial use is STRICTLY PROHIBITED without prior written permission**
- 📧 **You MUST contact the authors to obtain a commercial license**
- 💼 Please open an issue on GitHub or contact the authors for commercial licensing inquiries

See the [LICENSE](LICENSE) file for full details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

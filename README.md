# CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning

Todo:
I am checking the code and will update the code within 2 days, sorry for the delay (too busy for ICML 2026).
I will also update a blog for this paper :)  Code will be updated today!!!!!!!!


[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://openreview.net/forum?id=86IvZmY26S)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![State-of-the-Art](https://img.shields.io/badge/State--of--the--Art-Floorplanning-red.svg)](#results)

🏆 **The best-performing open-source learning-based floorplanning algorithm to date**, achieving state-of-the-art results on MCNC and GSRC benchmarks.

> **CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning**  
> Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang, Zhentao Tang, Mingxuan Yuan, Junchi Yan  
> *NeurIPS 2025*

## 🎯 Highlights

- 🥇 **State-of-the-Art Performance**: Achieves **12.9% improvement in wirelength** over previous best methods
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
├── CORE.py                # Main training script (parallel, recommended) ⭐
├── RL.py                   # RL training script (single process)
├── SA.py                   # Simulated Annealing baseline
├── Net.py                  # Neural network architectures (GNN + Transformer)
├── Makefile               # Build configuration
├── environment.yaml       # Conda environment specification
└── README.md              # This file
```

## Requirements

- Python 3.8
- PyTorch 1.4+ with CUDA support
- PyTorch Geometric 2.4+
- pybind11
- g++ (for compiling C++ code)

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate GPU_EDA
```

### 2. Compile C++ Code

The B\*-Tree floorplanner is implemented in C++ with Python bindings:

```bash
make
```

This generates the `tree` Python module (`tree.cpython-*.so`) that provides the B\*-Tree floorplanner functionality.

### 3. Verify Installation

```bash
python -c "import tree; print('CORE installed successfully!')"
```

## Usage

### Training with CORE (Recommended)

```bash
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
```

### Training with Simulated Annealing (Baseline)

```bash
python SA.py --circuit="n300" \
             --weight_hpwl=0.0 \
             --weight_area=0.5 \
             --gap_iter_update_temperature=1000
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--circuit` | Circuit name (n10, n30, n50, n100, n200, n300, ami33, ami49, etc.) | n300 |
| `--one_epsodic_length` | Episode length for RL | 100 |
| `--num_envs` | Number of parallel environments | 10 |
| `--device` | CUDA device ID (-1 for CPU) | 1 |
| `--L_weight` | Weight for HPWL (wirelength) | 1e-6 |
| `--A_weight` | Weight for area | 1e-6 |
| `--O_weight` | Weight for outbound penalty | 1e-6 |
| `--gamma` | Discount factor | 0.99 |
| `--batch_size` | Batch size for training | 128 |
| `--ent_coef` | Entropy coefficient | 0.01 |
| `--seed` | Random seed | 10 |
| `--add_res` | Add residual connections | False |
| `--clip-vloss` | Clip value loss | False |

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
| Previous SOTA | Baseline |
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.




# ALLM-Ab: Active Learning-Driven Antibody Optimization Using Fine-tuned Protein Language Models

This repository contains the code for ALLM-Ab, a multi-objective antibody optimization framework using protein language models.

1. **bindinggym_offline**: Evaluation of active learning in offline settings using BindingGYM dataset
2. **flexddg_online**: Implementation of active learning in online settings using Flex ddG

## Setup

### Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- PyGMO
- NumPy
- Pandas
- scikit-learn
- scipy
- Biopython
- tqdm
- peft
- ESM
- ablang2
- Easydict
- PyYAML
- PyGMO

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/ALLM-Ab.git
cd ALLM-Ab
```

2. Install required Python packages:
```bash
pip install numpy pandas torch torch_geometric tqdm scikit-learn scipy peft easydict PyYAML biopython
pip install fair-esm
pip install ablang2
pip install pygmo
```


3. [Flex ddG](https://github.com/Kortemme-Lab/flex_ddG_tutorial) installation is required for the flexddg_online component. (Optional)

### Data

The datasets can be obtained from BindingGYM:
```bash
git clone https://github.com/luwei0917/BindingGYM
```

## Usage

### bindinggym_offline

Example:
```bash
python al_run.py exps/outputs_ablang/0/greedy_0.0/dms_0_N-50_ini-1/config.yaml
```

### flexddg_online

Example:
```bash
cd flexddg_online
python al_run.py configs/5A12_dual/ablang2/greedy_dual.yaml
```

## Project Structure

- **bindinggym_offline**: Offline environment for protein binding simulation
- **flexddg_online**: Online analysis using Flex ddG
- **notebooks**: Jupyter notebooks for analysis
  - **reproduction_bindinggym.ipynb**: Notebook for reproducing bindinggym results
  - **reproduction_flexddg.ipynb**: Notebook for reproducing flexddg results
  - **analysis**: Analysis notebooks
- **results**: Results from experiments
  - **bindinggym_offline**: Results from bindinggym_offline
  - **flexddg_online**: Results from flexddg_online

## Citation

This work is currently under review.

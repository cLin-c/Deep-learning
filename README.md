### Features
Semi-Supervised Learning: Teacher-student architecture with exponential moving average updates
CLIP-Based Filtering: Intelligent unlabeled data filtering using CLIP vision-language model
Diffusion Model Augmentation: Advanced data augmentation using diffusion models
Multi-Scale Training: Support for various input sizes and multi-scale detection
Comprehensive Metrics: Built-in mAP calculation and performance tracking

### Table of Contents
1. Installation
2. Quick Start
3. Dataset Preparation
4. Training
5. Inference
6. Results
7. Contributing

### Installation
#### Prerequisites
Python 3.8+
CUDA 11.8+ (recommended)
8GB+ GPU memory (12GB+ for diffusion model version)

### Step 1: Create Environment
```markdown
```bash
conda create -n DSYM python=3.8 -y
conda activate DSYM
pip install torch torchvision
```

### Step 2: Install PyTorch
```markdown
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 3: Install Dependencies
bash# Core dependencies
pip install numpy opencv-python Pillow matplotlib seaborn pandas scipy scikit-learn tqdm PyYAML tensorboard

# CLIP support
pip install git+https://github.com/openai/CLIP.git

# Diffusion model support (for V4)
pip install diffusers transformers accelerate

# Optional: for acceleration and experiment tracking
pip install xformers wandb comet_ml thop



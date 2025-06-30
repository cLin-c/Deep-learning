DSYM环境配置和使用指南
1. 创建Conda虚拟环境
bash# 创建名为DSYM的conda虚拟环境，使用Python 3.8
conda create -n DSYM python=3.8 -y

# 激活环境
conda activate DSYM
2. 安装依赖包
2.1 安装PyTorch (根据你的CUDA版本选择)
bash# 如果有CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 如果有CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 如果只用CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch

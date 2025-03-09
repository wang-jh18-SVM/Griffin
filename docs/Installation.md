# Installation

Griffin leverages [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) and is optimized for CUDA 11.8. Follow these steps to set up your environment:

## Environment Setup
- Create and activate a Conda environment:
  ```bash
  conda create -n ENVNAME python=3.8 -y
  conda activate ENVNAME
  ```

## Install PyTorch
- Install PyTorch and torchvision:
  ```bash
  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  ```

## Set CUDA_HOME (Optional)
- Specify the CUDA path for GPU compilation (e.g., for spconv, mmdet3d):
  ```bash
  export CUDA_HOME=YOUR_CUDA_PATH/
  # Eg: export CUDA_HOME=/usr/local/cuda-11.8/
  ```

## Install Dependencies
- Install mmcv-full, mmdet, and mmsegmentation:
  ```bash
  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
  pip install mmdet==2.14.0 mmsegmentation==0.14.1
  ```
- Clone and install mmdet3d from source:
  ```bash
  git clone https://github.com/open-mmlab/mmdetection3d.git
  cd mmdetection3d
  git checkout v0.17.1
  pip install -v -e .
  ```

## Clone Griffin
- Set up the Griffin repository:
  ```bash
  git clone https://github.com/wang-jh18-SVM/Griffin.git griffin
  cd griffin
  pip install -r requirements.txt
  ```
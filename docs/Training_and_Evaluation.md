# Training and Evaluation

## Training

Training the cooperative model requires a three-step process using checkpoints from [BEVFormer](https://github.com/fundamentalvision/BEVFormer):

### 1. Vehicle-Side Model
- Train the vehicle-side model:
  ```bash
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh configs/CONFIG_FILE_VEHICLE NUM_GPUS
  ```

### 2. Drone-Side Model
- Train the drone-side model:
  ```bash
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_DRONE NUM_GPUS
  ```
- Modify the parameters in CONFIG_FILE_DRONE:
  ```python
  save_track_query=True
  ```
- Run inference to save track queries:
  ```bash
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_eval.sh CONFIG_FILE_DRONE CHECKPOINT NUM_GPUS
  ```

### 3. Cooperative Model
- Train using saved drone-side track queries:
  ```bash
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_COOP NUM_GPUS
  ```

## Evaluation

Evaluate models using the provided checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm):
```
└── ckpts
    ├── griffin_50scenes_25m
    │   ├── cooperative.pth  (md5: 03479d564e1b4a2cf3a58ae08899ff72)
    │   ├── drone-side.pth   (md5: d6cb7b6e5914bf807a1faf8c5b69600b)
    │   └── vehicle-side.pth (md5: 256d80f0a2791402ac7fa217d13f733b)
    ├── griffin_50scenes_40m
    │   ├── cooperative.pth  (md5: f7f9d91c849a7c419dce17a547917779)
    │   ├── drone-side.pth   (md5: 1845bc94a7f52605a53601f48cc96072)
    │   └── vehicle-side.pth (md5: 961187ff3a23bc94aa2d216ce2f60df8)
    └── griffin_100scenes_random
        ├── cooperative.pth  (md5: 6717b83e9e9f2ce1449cd09ebb2c4a9c)
        ├── drone-side.pth   (md5: fccb599faa8d3b77e99d5af35a9b8ae8)
        └── vehicle-side.pth (md5: 64f3bcd5af1bfc908ae1e1fae6ce4a3a)
```

### Vehicle-Side
```bash
./tools/dist_eval.sh CONFIG_FILE_VEHICLE CHECKPOINT_VEHICLE 1
```

### Drone-Side
```bash
./tools/dist_eval.sh CONFIG_FILE_DRONE CHECKPOINT_DRONE 1
```

### Cooperative
```bash
./tools/dist_eval.sh CONFIG_FILE_COOP CHECKPOINT_COOP 1
```
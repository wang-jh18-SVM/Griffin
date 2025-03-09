# Dataset Preparation

Griffin is split into three subsets based on UAV cruising altitude:
- **Griffin-Random**: 20–60 meters (104 scenes)
- **Griffin-25m**: 25 ± 2 meters (47 scenes)
- **Griffin-40m**: 40 ± 2 meters (54 scenes)

Each of the 205 scenes lasts ~15 seconds (~150 frames), totaling over 30,000 samples and 275,000 images. The dataset supports both KITTI (ego-centric) and NuScenes (global reference) formats.

## Download Dataset
- Get Griffin from [Baidu Netdisk](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm) and organize as follows:
  ```
  └── datasets
      ├── griffin_50scenes_25m (md5: 391e6ba2ddc97615d092cd9cb04d21e0)
      │   └── griffin-release
      │       ├── vehicle-side
      │       └── drone-side
      ├── griffin_50scenes_40m (md5: adba8dc3ee2e1f5635b60f48559252a2)
      │   └── griffin-release
      │       ├── vehicle-side
      │       └── drone-side
      └── griffin_100scenes_random (md5: 96c9bf0a3ade9490b390d53dcff06e26)
          └── griffin-release
              ├── vehicle-side
              └── drone-side
  ```

## Convert to NuScenes Format
- Transform KITTI-style raw data to NuScenes style:
  ```bash
  bash tools/griffin_converter.sh griffin_50scenes_25m
  bash tools/griffin_converter.sh griffin_50scenes_40m
  bash tools/griffin_converter.sh griffin_100scenes_random
  ```
- Resulting structure (e.g., Griffin-25m):
  ```
  ├── data
  │   ├── infos
  │   │   └── griffin_50scenes_25m
  │   │       ├── cooperative
  │   │       │   ├── griffin_infos_train.pkl
  │   │       │   └── griffin_infos_val.pkl
  │   │       ├── drone-side
  │   │       └── vehicle-side
  │   └── split_datas
  │       └── griffin_50scenes_25m.json
  └── datasets
      └── griffin_50scenes_25m
          ├── griffin-nuscenes
          │   ├── cooperative
          │   ├── drone-side
          │   ├── early-fusion
          │   └── vehicle-side
          └── griffin-release
              ├── vehicle-side
              └── drone-side
  ```
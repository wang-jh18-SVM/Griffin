# Dataset Preparation

Griffin is split into three subsets based on UAV cruising altitude:
- **Griffin-Random**: 20–60 meters (104 scenes)
- **Griffin-25m**: 25 ± 2 meters (47 scenes)
- **Griffin-40m**: 40 ± 2 meters (54 scenes)

Each of the 205 scenes lasts ~15 seconds (~150 frames), totaling over 30,000 samples and 275,000 images. The dataset supports both KITTI (ego-centric) and NuScenes (global reference) formats.

## Download Dataset
- Get Griffin from [Baidu Netdisk](https://pan.baidu.com/s/1NDgsuHB-QPRiROV73NRU5g?pwd=u3cm) or [Hugging Face](https://huggingface.co/datasets/wjh-svm/Griffin) and organize as follows: (showing griffin_50scenes_25m as an example, other datasets have similar structures)
  ```
  └── datasets
      ├── griffin_50scenes_25m
      │   └── griffin-release
      │       ├── vehicle-side
      │       │   ├── calib
      │       │   ├── camera
      │       │   │   ├── back
      │       │   │   ├── front
      │       │   │   ├── instance_back
      │       │   │   ├── instance_front
      │       │   │   ├── instance_left
      │       │   │   ├── instance_right
      │       │   │   ├── left
      │       │   │   └── right
      │       │   ├── label
      │       │   ├── lidar
      │       │   │   └── lidar_top
      │       │   ├── pose
      │       │   └── scene_infos.json
      │       └── drone-side
      │           ├── calib
      │           ├── camera
      │           │   ├── back
      │           │   ├── bottom
      │           │   ├── front
      │           │   ├── instance_back
      │           │   ├── instance_bottom
      │           │   ├── instance_front
      │           │   ├── instance_left
      │           │   ├── instance_right
      │           │   ├── left
      │           │   └── right
      │           ├── label
      │           ├── pose
      │           └── scene_infos.json
      ├── griffin_50scenes_40m
      └── griffin_100scenes_random
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
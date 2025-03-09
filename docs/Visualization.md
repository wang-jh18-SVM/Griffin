# Visualization

Visualize Griffin data in two formats:

## KITTI Style (Single-Side Ground Truth)
- Vehicle-side:
  ```bash
  python tools/griffin_data_converter/visual_kitti.py datasets/griffin_50scenes_25m/griffin-release/vehicle-side/
  ```
- Drone-side:
  ```bash
  python tools/griffin_data_converter/visual_kitti.py datasets/griffin_50scenes_25m/griffin-release/drone-side/
  ```

## NuScenes Style (Cooperative Results)
- Visualize cooperative data and detections:
  ```bash
  python tools/analysis_tools/visual_griffin.py --dataroot datasets/griffin_50scenes_25m/griffin-nuscenes/early-fusion --out_folder result_vis/griffin_50scenes_25m
  ```

[![Griffin Demo Video](./figure/label_visualization.png)](./video/Griffin_r1200_10fps_1_3Mbps.mp4)
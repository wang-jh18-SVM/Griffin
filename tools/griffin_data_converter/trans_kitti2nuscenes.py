import os
import json
import random
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from nuscenes.eval.common.utils import Quaternion
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List
from data_utils import (
    load_calibration,
    load_pose,
    load_label,
    load_scene_infos,
    ObjectData,
    PoseData,
    CalibrationParams,
)
from space_utils import trans_object_vs_ego, trans_ego_vs_ENU
import argparse
from nuscenes.nuscenes import NuScenes
import copy
import multiprocessing
from functools import partial

class GriffinKittiToNuScenesConverter:
    PC_RANGE = {
        "x": [-50, 50],
        "y": [-50, 50],
    }
    MIN_SCENE_LENGTH = 10

    def __init__(self, source_dir: str, target_dir: str, side: str):
        self.source_dir = Path(source_dir).absolute()
        self.target_dir = Path(target_dir).absolute()
        self.version = "v1.0-trainval"
        self.side = side
        self.num_workers = multiprocessing.cpu_count()

        if self.side == "drone":
            self.sensor_types = {
                'front': 'cam',
                'back': 'cam',
                'left': 'cam',
                'right': 'cam',
                'bottom': 'cam',
                # 'lidar_front': 'lidar',
                # 'lidar_bottom': 'lidar',
                'top': 'lidar',
            }
        elif self.side == "vehicle":
            self.sensor_types = {
                'front': 'cam',
                'back': 'cam',
                'left': 'cam',
                'right': 'cam',
                'top': 'lidar',
            }
        elif self.side == "cooperative":
            self.sensor_types = {
                "front": "cam",
                "back": "cam",
                "left": "cam",
                "right": "cam",
                "top": "lidar",
                # "front_air": "cam_air",
                # "back_air": "cam_air",
                # "left_air": "cam_air",
                # "right_air": "cam_air",
                # "bottom_air": "cam_air",
                "top_air": "lidar_air",
            }
            self.source_dir_air = os.path.join(self.source_dir, "drone-side")
            self.source_dir = os.path.join(self.source_dir, "vehicle-side")
        else:
            raise ValueError(f"Invalid side: {self.side}")

        self.obj_types = {
            12: "pedestrian",
            # 13: "rider",
            14: "car",
            15: "truck",
            16: "bus",
            # 17: "train",
            18: "motorcycle",
            19: "bicycle",
        }
        self.obj_type_mapping = {
            "pedestrian": "pedestrian",
            "car": "car",
            "truck": "car",
            "bus": "car",
            "motorcycle": "bicycle",
            "bicycle": "bicycle",
        }
        self.categories = ["pedestrian", "car", "bicycle"]

        self.frame_list = [
            f.split('.')[0]
            for f in os.listdir(os.path.join(self.source_dir, 'pose'))
            if f.endswith('.json')
        ]
        self.frame_list.sort()
        self.timestamp_list = [
            self._frame_number_to_nuscenes_timestamp(f) for f in self.frame_list
        ]

        # Cache for processed data
        self.calibration_data: Dict[str, CalibrationParams] = {}
        self.ego_poses: Dict[str, PoseData] = {}
        self.annotations: Dict[str, List[ObjectData]] = {}
        if self.side == "cooperative":
            self.calibration_data_air: Dict[str, CalibrationParams] = {}
            self.ego_poses_air: Dict[str, PoseData] = {}
            self.annotations_air: Dict[str, List[ObjectData]] = {}

        # Initialize metadata structures
        self.metadata = {
            'attribute': [],
            'calibrated_sensor': [],
            'category': [],
            'ego_pose': [],
            'instance': [],
            'log': [],
            'map': [],
            'sample_annotation': [],
            'sample_data': [],
            'sample': [],
            'scene': [],
            'sensor': [],
            'visibility': [],
        }

    def create_directory_structure(self):
        """Create the required directory structure"""
        dirs = [
            self.target_dir / 'samples',
            self.target_dir / 'maps',
            self.target_dir / self.version,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def copy_sensor_data(self):
        """Copy sensor data files to the target directory"""
        for sensor_name, sensor_type in self.sensor_types.items():
            if sensor_type == 'cam':
                source_sensor_dir = os.path.join(self.source_dir, 'camera', sensor_name)
                target_sensor_dir = os.path.join(
                    self.target_dir, 'samples', f"{sensor_type}_{sensor_name}".upper()
                )
            elif self.side == "early-fusion" and sensor_type == 'cam_air':
                sensor_name_air = sensor_name.split('_')[0]
                source_sensor_dir = os.path.join(
                    self.source_dir_air, 'camera', sensor_name_air
                )
                target_sensor_dir = os.path.join(
                    self.target_dir,
                    'samples',
                    f"{sensor_type.split('_')[0]}_{sensor_name}".upper(),
                )
            else:
                continue

            if not os.path.exists(target_sensor_dir):
                print(f"Linking {source_sensor_dir} data to {target_sensor_dir}...")
                os.symlink(source_sensor_dir, target_sensor_dir)
            else:
                print(f"Sensor data already exists at {target_sensor_dir}")

    def load_metadata(self):
        """Load metadata from source"""
        for sensor_name, sensor_type in self.sensor_types.items():
            if sensor_type == 'cam':
                self.calibration_data[sensor_name] = load_calibration(
                    self.source_dir, sensor_name
                )
            elif sensor_name == 'top':
                if self.side == "drone":
                    self.calibration_data[sensor_name] = load_calibration(
                        self.source_dir, 'lidar_virtual'
                    )
                elif self.side == "vehicle" or self.side == "cooperative":
                    self.calibration_data[sensor_name] = load_calibration(
                        self.source_dir, 'lidar_top'
                    )
            elif self.side == "cooperative" and "_air" in sensor_name:
                self.calibration_data[sensor_name] = load_calibration(
                    self.source_dir_air, 'lidar_virtual'
                )
            else:
                raise ValueError(f"Invalid sensor: {sensor_name}")

        for frame in tqdm(self.frame_list, desc="Loading metadata by frame"):
            self.ego_poses[frame] = load_pose(self.source_dir, frame)
            self.annotations[frame] = load_label(self.source_dir, frame)
            if self.side == "cooperative":
                self.ego_poses_air[frame] = load_pose(self.source_dir_air, frame)
                self.annotations_air[frame] = load_label(self.source_dir_air, frame)

    def check_invalid_frames(self):
        """
        Project all annotations to the ego top lidar coordinate system,
        only keep the frames that simultaneously have annotations within PC_RANGE in ego vehicle and ego drone coordinate system,
        as well as have annotations from air side within PC_RANGE in ego vehicle coordinate system.
        Then check if the frames are continuous long enough.
        """
        assert self.side == "cooperative", "Only cooperative side is supported"

        invalid_frames = {}  # {idx: [frame, timestamp]}
        for idx, frame in tqdm(enumerate(self.frame_list), total=len(self.frame_list)):
            # check if the frame is valid in ego vehicle coordinate system
            is_veh_frame_valid = False
            top_lidar2ego_matrix = self.calibration_data['top'].extrinsic
            ego2lidar_matrix = np.linalg.inv(top_lidar2ego_matrix)
            for ann in self.annotations[frame]:
                obj2ego_matrix, _ = trans_object_vs_ego(ann)
                obj2lidar_matrix = ego2lidar_matrix @ obj2ego_matrix
                t_obj2lidar = obj2lidar_matrix[:3, 3]

                if (
                    self.PC_RANGE["x"][0] <= t_obj2lidar[0] <= self.PC_RANGE["x"][1]
                    and self.PC_RANGE["y"][0] <= t_obj2lidar[1] <= self.PC_RANGE["y"][1]
                ):
                    is_veh_frame_valid = True
                    break

            # check if the frame is valid in ego drone coordinate system
            is_air_frame_valid = False
            top_air_lidar2air_matrix = self.calibration_data['top_air'].extrinsic
            air2air_lidar_matrix = np.linalg.inv(top_air_lidar2air_matrix)
            for ann in self.annotations_air[frame]:
                obj2air_matrix, _ = trans_object_vs_ego(ann)
                obj2air_lidar_matrix = air2air_lidar_matrix @ obj2air_matrix
                t_obj2air_lidar = obj2air_lidar_matrix[:3, 3]

                if (
                    self.PC_RANGE["x"][0] <= t_obj2air_lidar[0] <= self.PC_RANGE["x"][1]
                    and self.PC_RANGE["y"][0]
                    <= t_obj2air_lidar[1]
                    <= self.PC_RANGE["y"][1]
                ):
                    is_air_frame_valid = True
                    break

            # check if the frame is valid in ego vehicle coordinate system, including annotations from ego vehicle and ego drone together
            is_coop_frame_valid = is_veh_frame_valid
            if not is_coop_frame_valid:
                pose_ego = self.ego_poses[frame]
                pose_air = self.ego_poses_air[frame]
                for ann in self.annotations_air[frame]:
                    obj2air_matrix, _ = trans_object_vs_ego(ann)
                    air2ENU_matrix, _ = trans_ego_vs_ENU(pose_air)
                    _, ENU2ego_matrix = trans_ego_vs_ENU(pose_ego)

                    obj2lidar_matrix = (
                        ego2lidar_matrix
                        @ ENU2ego_matrix
                        @ air2ENU_matrix
                        @ obj2air_matrix
                    )
                    t_obj2lidar = obj2lidar_matrix[:3, 3]

                    if (
                        self.PC_RANGE["x"][0] <= t_obj2lidar[0] <= self.PC_RANGE["x"][1]
                        and self.PC_RANGE["y"][0]
                        <= t_obj2lidar[1]
                        <= self.PC_RANGE["y"][1]
                    ):
                        is_frame_valid = True
                        break

            if (
                not is_veh_frame_valid
                or not is_air_frame_valid
                or not is_coop_frame_valid
            ):
                invalid_frames[idx] = [frame, self.timestamp_list[idx]]
        print(f"Found {len(invalid_frames)} empty frames")

        # Checking continuous frames
        non_empty_indices = [
            idx for idx in range(len(self.frame_list)) if idx not in invalid_frames
        ]
        non_empty_indices.sort()

        # Find all continuous segments
        segments = []
        if non_empty_indices:
            current_frame_segment_idxs = [non_empty_indices[0]]
            for frame_idx in non_empty_indices[1:]:
                if frame_idx - 1 in non_empty_indices and (
                    int(self.frame_list[frame_idx])
                    - int(self.frame_list[frame_idx - 1])
                    == 1
                ):
                    current_frame_segment_idxs.append(frame_idx)
                else:
                    segments.append(current_frame_segment_idxs)
                    current_frame_segment_idxs = [frame_idx]
            segments.append(current_frame_segment_idxs)

        # Mark short segments as invalid
        for seg in segments:
            if len(seg) < self.MIN_SCENE_LENGTH:
                for idx in seg:
                    if idx not in invalid_frames:
                        invalid_frames[idx] = [
                            self.frame_list[idx],
                            self.timestamp_list[idx],
                        ]

        print(f"Total invalid frames after continuity check: {len(invalid_frames)}")
        return invalid_frames

    def remove_invalid_frames(self, invalid_frames: dict):
        """Remove invalid frames"""
        # reverse order to avoid index shifting during popping
        for idx in sorted(invalid_frames.keys(), reverse=True):
            frame = invalid_frames[idx][0]
            timestamp = invalid_frames[idx][1]

            assert frame == self.frame_list.pop(idx), "Frame mismatch"
            assert timestamp == self.timestamp_list.pop(idx), "Timestamp mismatch"

            self.ego_poses.pop(frame)
            self.annotations.pop(frame)
            if self.side == "cooperative":
                self.ego_poses_air.pop(frame)
                self.annotations_air.pop(frame)

    def generate_metadata(self):
        """Generate all required metadata files"""
        self.generate_sensor_metadata()
        self.generate_calibrated_sensor_metadata()
        self.generate_category_metadata()
        self.generate_ego_pose_metadata()
        self.generate_sample_metadata()
        self.generate_sample_data_metadata()
        self.generate_instance_metadata()
        self.generate_sample_annotation_metadata()
        self.generate_scene_metadata()
        self.generate_visibility_metadata()

        self.generate_placeholder_metadata()

        self._verify_metadata_integrity()

    def generate_sensor_metadata(self):
        """Generate sensor.json metadata"""
        print("Generating sensor metadata...")
        for sensor_name, sensor_type in self.sensor_types.items():
            sensor_data = {
                'token': self._generate_token(f'sensor_{sensor_name}'),
                'channel': f"{sensor_type.split('_')[0]}_{sensor_name}".upper(),
                'modality': sensor_type.split('_')[0].upper(),
            }
            self.metadata['sensor'].append(sensor_data)

    def generate_calibrated_sensor_metadata(self):
        """Generate calibrated_sensor.json metadata"""
        print("Generating calibrated sensor metadata...")

        top_lidar2ego_matrix = self.calibration_data['top'].extrinsic
        if self.side == "cooperative":
            top_air_lidar2air_matrix = self.calibration_data['top_air'].extrinsic

        for sensor_name, calib_data in self.calibration_data.items():
            sensor2ego_matrix = calib_data.extrinsic
            if self.side != "cooperative" or "_air" not in sensor_name:
                lidar2sensor_matrix = (
                    np.linalg.inv(sensor2ego_matrix) @ top_lidar2ego_matrix
                )
            else:
                lidar2sensor_matrix = (
                    np.linalg.inv(sensor2ego_matrix) @ top_air_lidar2air_matrix
                )
            sensor2lidar_matrix = np.linalg.inv(lidar2sensor_matrix)

            calibrated_sensor = {
                'token': self._generate_token(f'calibrated_sensor_{sensor_name}'),
                'sensor_token': self._get_sensor_token(sensor_name),
                'translation': sensor2ego_matrix[:3, 3].tolist(),
                'rotation': Quaternion(
                    matrix=sensor2ego_matrix[:3, :3]
                ).elements.tolist(),
                'camera_intrinsic': (
                    calib_data.intrinsic.tolist()
                    if sensor_name != 'top' and sensor_name != 'top_air'
                    else []
                ),
                'camera_distortion': [],  # Placeholder
                'lidar2cam_translation': lidar2sensor_matrix[:3, 3].tolist(),
                'lidar2cam_rotation': Quaternion(
                    matrix=lidar2sensor_matrix[:3, :3]
                ).elements.tolist(),
                'sensor2lidar_translation': sensor2lidar_matrix[:3, 3].tolist(),
                'sensor2lidar_rotation': Quaternion(
                    matrix=sensor2lidar_matrix[:3, :3]
                ).elements.tolist(),
            }
            self.metadata['calibrated_sensor'].append(calibrated_sensor)

    def generate_category_metadata(self):
        """Generate category.json metadata"""
        print("Generating category metadata...")
        for category in self.categories:
            category = {
                'token': self._generate_token(f'category_{category}'),
                'name': category,
                'description': f'{category} category',
            }
            self.metadata['category'].append(category)

    def generate_ego_pose_metadata(self):
        """Generate ego_pose.json metadata"""
        print("Generating ego pose metadata...")
        for idx, frame in enumerate(self.frame_list):
            timestamp = self.timestamp_list[idx]
            pose_data = self.ego_poses[frame]
            T_ego_to_ENU, _ = trans_ego_vs_ENU(pose_data)

            ego_pose = {
                'token': self._generate_token(f'ego_pose_{timestamp}'),
                'timestamp': timestamp,
                'rotation': Quaternion(matrix=T_ego_to_ENU[:3, :3]).elements.tolist(),
                'translation': T_ego_to_ENU[:3, 3].tolist(),
            }

            if self.side == "cooperative":
                pose_data_air = self.ego_poses_air[frame]
                T_air_to_ENU, _ = trans_ego_vs_ENU(pose_data_air)
                # ego_pose_air = {
                #     'token': self._generate_token(f'air_pose_{timestamp}'),
                #     'timestamp': timestamp,
                #     'rotation': Quaternion(
                #         matrix=T_air_to_ENU[:3, :3]
                #     ).elements.tolist(),
                #     'translation': T_air_to_ENU[:3, 3].tolist(),
                # }
                # self.metadata['ego_pose'].append(ego_pose_air)

                ego_pose['air_rotation'] = Quaternion(
                    matrix=T_air_to_ENU[:3, :3]
                ).elements.tolist()
                ego_pose['air_translation'] = T_air_to_ENU[:3, 3].tolist()

            self.metadata['ego_pose'].append(ego_pose)

    def generate_sample_metadata(self):
        """Generate sample.json metadata"""
        print("Generating sample metadata...")

        prev_token = ''
        scene_num = 0
        for idx, frame in enumerate(self.frame_list):
            timestamp = self.timestamp_list[idx]
            is_next_sample_continuous = (idx < len(self.frame_list) - 1) and (
                int(self.frame_list[idx + 1]) - int(frame) == 1
            )
            # is_next_sample_continuous = True

            next_token = (
                self._generate_token(f'sample_{self.timestamp_list[idx + 1]}')
                if is_next_sample_continuous
                else ''
            )

            sample = {
                'token': self._generate_token(f'sample_{timestamp}'),
                'timestamp': timestamp,
                'prev': prev_token,
                'next': next_token,
                'scene_token': self._generate_token(f'scene_{scene_num}'),
                'frame_idx': idx,
                'ego_pose_token': self._generate_token(f'ego_pose_{timestamp}'),
            }
            # if self.side == "cooperative":
            #     sample['air_pose_token'] = self._generate_token(f'air_pose_{timestamp}')
            self.metadata['sample'].append(sample)

            if is_next_sample_continuous:
                prev_token = sample['token']
            else:
                prev_token = ''
                scene_num += 1

    def generate_sample_data_metadata(self):
        """Generate sample_data.json metadata"""
        print("Generating sample data metadata...")

        for sensor_name, sensor_type in self.sensor_types.items():
            prev_token = ''
            scene_num = 0
            for idx, frame in enumerate(self.frame_list):
                timestamp = self.timestamp_list[idx]

                is_next_sample_continuous = (idx < len(self.frame_list) - 1) and (
                    int(self.frame_list[idx + 1]) - int(frame) == 1
                )
                # is_next_sample_continuous = True

                next_token = (
                    self._generate_token(
                        f'data_{sensor_name}_{self.timestamp_list[idx + 1]}'
                    )
                    if is_next_sample_continuous
                    else ''
                )

                sample_data = {
                    'token': self._generate_token(f'data_{sensor_name}_{timestamp}'),
                    'sample_token': self._generate_token(f'sample_{timestamp}'),
                    'ego_pose_token': (
                        self._generate_token(f'ego_pose_{timestamp}')
                        # if "_air" not in sensor_name
                        # else self._generate_token(f'air_pose_{timestamp}')
                    ),
                    'calibrated_sensor_token': self._get_calibrated_sensor_token(
                        sensor_name
                    ),
                    'timestamp': timestamp,
                    'prev': prev_token,
                    'next': next_token,
                    'is_key_frame': True,
                }

                if sensor_type == 'cam':  # TODO: add instance segmentation and depth
                    sample_data['filename'] = os.path.join(
                        'samples',
                        f"{sensor_type.split('_')[0]}_{sensor_name}".upper(),
                        f"{frame}.png",
                    )
                    sample_data['fileformat'] = 'png'
                    sample_data['width'] = 1920
                    sample_data['height'] = 1080
                elif sensor_name == 'top':
                    sample_data['filename'] = ''
                    sample_data['fileformat'] = ''
                    sample_data['width'] = 0
                    sample_data['height'] = 0

                self.metadata['sample_data'].append(sample_data)

                if is_next_sample_continuous:
                    prev_token = sample_data['token']
                else:
                    prev_token = ''
                    scene_num += 1

    def generate_instance_metadata(self):
        """Generate instance.json metadata"""
        print("Generating instance metadata...")

        unique_instances = {}
        for idx, frame in enumerate(self.frame_list):
            timestamp = self.timestamp_list[idx]
            annotations = copy.deepcopy(self.annotations[frame])

            if self.side == "cooperative":
                annotations_air = self.annotations_air[frame]
                air_ann_id_list = [ann_air.id for ann_air in annotations_air]
                ego_ann_id_list = [ann.id for ann in annotations]
                only_air_ann_id_list = list(set(air_ann_id_list) - set(ego_ann_id_list))

                for ann_air in annotations_air:
                    if ann_air.id in only_air_ann_id_list:
                        annotations.append(ann_air)  # TODO: filter out ego vehicle

            for ann in annotations:
                if ann.id not in unique_instances:
                    unique_instances[ann.id] = {
                        'token': self._generate_token(f'instance_{ann.id}'),
                        'category_token': self._get_category_token(
                            self.obj_type_mapping[ann.type.lower()]
                        ),
                        'nbr_annotations': 1,
                        'first_annotation_token': self._generate_token(
                            f'annotation_{timestamp}_{ann.id}'
                        ),
                        'last_annotation_token': self._generate_token(
                            f'annotation_{timestamp}_{ann.id}'
                        ),
                        'annonation_tokens': [
                            self._generate_token(f'annotation_{timestamp}_{ann.id}'),
                        ],
                    }
                else:
                    unique_instances[ann.id]['nbr_annotations'] += 1
                    unique_instances[ann.id]['last_annotation_token'] = (
                        self._generate_token(f'annotation_{timestamp}_{ann.id}')
                    )
                    unique_instances[ann.id]['annonation_tokens'].append(
                        self._generate_token(f'annotation_{timestamp}_{ann.id}')
                    )

        for instance_id in unique_instances:
            self.metadata['instance'].append(unique_instances[instance_id])

    def generate_sample_annotation_metadata(self):
        """Generate sample_annotation.json metadata"""
        print("Generating sample annotation metadata...")

        for idx, frame in enumerate(
            tqdm(self.frame_list, desc="Generating sample annotation metadata by frame")
        ):
            timestamp = self.timestamp_list[idx]
            annotations: List[ObjectData] = copy.deepcopy(self.annotations[frame])
            ego_pose: PoseData = self.ego_poses[frame]
            T_ego_to_ENU, T_ENU_to_ego = trans_ego_vs_ENU(ego_pose)

            if self.side == "cooperative":
                annotations_air: List[ObjectData] = self.annotations_air[frame]
                ego_pose_air: PoseData = self.ego_poses_air[frame]
                T_air_to_ENU, _ = trans_ego_vs_ENU(ego_pose_air)

                air_ann_id_list = [ann_air.id for ann_air in annotations_air]
                ego_ann_id_list = [ann.id for ann in annotations]
                common_ann_id_list = list(set(air_ann_id_list) & set(ego_ann_id_list))
                only_air_ann_id_list = list(set(air_ann_id_list) - set(ego_ann_id_list))
                # if len(only_air_ann_id_list) > 0:
                #     print(f"Only air annotations: {only_air_ann_id_list}")

                for ann_air in annotations_air:
                    if ann_air.id in only_air_ann_id_list:
                        annotations.append(ann_air)

            for ann in annotations:
                if self.side == "cooperative" and ann.id in only_air_ann_id_list:
                    T_ann_to_air, _ = trans_object_vs_ego(ann)
                    T_ann_to_ENU = T_air_to_ENU @ T_ann_to_air

                    # Filter out ego vehicle in air annotations
                    T_ann_to_ego = T_ENU_to_ego @ T_ann_to_ENU
                    t_ann_to_ego = T_ann_to_ego[:3, 3]
                    if np.linalg.norm(t_ann_to_ego[:2]) < 0.1:
                        # print(
                        #     f"Frame {frame}: obj {ann.id} is too close to ego vehicle"
                        # )
                        continue
                else:
                    T_ann_to_ego, _ = trans_object_vs_ego(ann)
                    T_ann_to_ENU = T_ego_to_ENU @ T_ann_to_ego

                vis_rate = ann.visibility
                vis_side = self.side
                if self.side == "cooperative":
                    if ann.id in common_ann_id_list:
                        vis_rate_air = annotations_air[
                            air_ann_id_list.index(ann.id)
                        ].visibility
                        vis_rate = max(vis_rate, vis_rate_air)
                    elif ann.id in only_air_ann_id_list:
                        vis_rate = 0
                        vis_side = "drone"
                    else:
                        vis_side = "vehicle"

                vis_level = (
                    1
                    if vis_rate < 0.4
                    else (2 if vis_rate < 0.6 else (3 if vis_rate < 0.8 else 4))
                )

                annotation = {
                    'token': self._generate_token(f'annotation_{timestamp}_{ann.id}'),
                    'sample_token': self._generate_token(f'sample_{timestamp}'),
                    'instance_token': self._generate_token(f'instance_{ann.id}'),
                    'visibility_token': self._generate_token(f'visibility_{vis_level}'),
                    'attribute_tokens': [
                        self._generate_token('attribute_default')
                    ],  # Placeholder
                    'category_token': self._get_category_token(
                        self.obj_type_mapping[ann.type.lower()]
                    ),
                    # 'size': [ann.l, ann.w, ann.h], # nuscenes format is w,l,h
                    'size': [ann.w, ann.l, ann.h],
                    'translation': T_ann_to_ENU[:3, 3].tolist(),
                    'rotation': Quaternion(
                        matrix=T_ann_to_ENU[:3, :3]
                    ).elements.tolist(),
                    'num_lidar_pts': 1,
                    'num_radar_pts': 1,
                    'prev': self._get_instance_prev_anno_token(
                        self._generate_token(f'instance_{ann.id}'),
                        self._generate_token(f'annotation_{timestamp}_{ann.id}'),
                    ),
                    'next': self._get_instance_next_anno_token(
                        self._generate_token(f'instance_{ann.id}'),
                        self._generate_token(f'annotation_{timestamp}_{ann.id}'),
                    ),
                    "id": ann.id,
                    "vis_side": vis_side,
                }
                self.metadata['sample_annotation'].append(annotation)

    def generate_scene_metadata(self):
        """Generate scene.json metadata"""
        print("Generating scene metadata...")
        scene_infos, frame2scene = load_scene_infos(self.source_dir)

        prev_scene_token = ''
        scene_num = 0
        for idx, sample in enumerate(self.metadata['sample']):
            if sample['scene_token'] != prev_scene_token:
                if idx > 0:
                    self.metadata['scene'].append(scene)
                    scene_num += 1

                scene_info = scene_infos[frame2scene[self.frame_list[idx]]]
                scene = {
                    'token': self._generate_token(f'scene_{scene_num}'),
                    'name': f'scene-{scene_num:04}-{scene_info["name"]}',
                    'description': scene_info["info"]["weather"],
                    'log_token': self._generate_token('log_default'),
                    'nbr_samples': 1,
                    'first_sample_token': sample['token'],
                    'last_sample_token': sample['token'],
                }
                prev_scene_token = sample['scene_token']
            else:
                scene['nbr_samples'] += 1
                scene['last_sample_token'] = sample['token']

            if idx == len(self.metadata['sample']) - 1:
                self.metadata['scene'].append(scene)

    def generate_visibility_metadata(self):
        """Generate visibility.json metadata"""
        print("Generating visibility metadata...")
        vis_description = {
            1: 'visibility of whole object is between 0 and 40%',
            2: 'visibility of whole object is between 40 and 60%',
            3: 'visibility of whole object is between 60 and 80%',
            4: 'visibility of whole object is between 80 and 100%',
        }
        for vis_level, vis_desc in vis_description.items():
            visibility_data = {
                'token': self._generate_token(f'visibility_{vis_level}'),
                'level': vis_level,
                'description': vis_desc,
            }
            self.metadata['visibility'].append(visibility_data)

    def generate_placeholder_metadata(self):
        """Generate placeholder metadata for missing information"""
        # Generate attribute.json
        print("Generating attribute placeholder metadata...")
        default_attribute = {
            'token': self._generate_token('attribute_default'),
            'name': '',
            'description': 'Default attribute as empty',
        }
        self.metadata['attribute'].append(default_attribute)

        # Generate log.json
        print("Generating log placeholder metadata...")
        log_data = {
            'token': self._generate_token('log_default'),
            'logfile': '',
            'vehicle': 'drone',
            'date_captured': datetime(2025, 1, 1, 9, 0, 0).strftime('%Y-%m-%d'),
            'location': 'carla',
        }
        self.metadata['log'].append(log_data)

        # Generate map.json
        print("Generating map placeholder metadata...")
        map_data = {
            'token': self._generate_token('map_default'),
            'filename': '',
            'category': 'semantic_prior',
            'log_tokens': [self._generate_token('log_default')],
        }
        self.metadata['map'].append(map_data)

    # Helper methods
    def _generate_token(self, base: str) -> str:
        """Generate a unique token based on input string"""
        # import hashlib

        # return hashlib.sha1(base.encode()).hexdigest()
        return base

    def _get_sensor_token(self, sensor_name: str) -> str:
        """Get sensor token by sensor name"""
        for sensor in self.metadata['sensor']:
            sensor_type = self.sensor_types[sensor_name]
            if (
                sensor['channel']
                == f"{sensor_type.split('_')[0]}_{sensor_name}".upper()
            ):
                return sensor['token']
        print(f"Warning: Sensor {sensor_name} not found")
        return ''

    def _get_calibrated_sensor_token(self, sensor_name: str) -> str:
        """Get calibrated sensor token by sensor name"""
        for sensor in self.metadata['calibrated_sensor']:
            if sensor['sensor_token'] == self._get_sensor_token(sensor_name):
                return sensor['token']
        print(f"Warning: Calibrated sensor {sensor_name} not found")
        return ''

    def _get_category_token(self, category_name: str) -> str:
        """Get category token by category name"""
        for category in self.metadata['category']:
            if category['name'] == category_name:
                return category['token']
        print(f"Warning: Category {category_name} not found")
        return ''

    def _get_instance_prev_anno_token(
        self, instance_token: str, curr_anno_token: str
    ) -> str:
        """Get previous instance token by instance id"""
        for instance in self.metadata['instance']:
            if instance['token'] == instance_token:
                curr_idx = instance['annonation_tokens'].index(curr_anno_token)
                if curr_idx > 0:
                    return instance['annonation_tokens'][curr_idx - 1]
                else:
                    return ''
        print(f"Warning: Instance {instance_token} not found")
        return ''

    def _get_instance_next_anno_token(
        self, instance_token: str, curr_anno_token: str
    ) -> str:
        """Get next instance token by instance id"""
        for instance in self.metadata['instance']:
            if instance['token'] == instance_token:
                curr_idx = instance['annonation_tokens'].index(curr_anno_token)
                if curr_idx < len(instance['annonation_tokens']) - 1:
                    return instance['annonation_tokens'][curr_idx + 1]
                else:
                    return ''
        print(f"Warning: Instance {instance_token} not found")
        return ''

    def _frame_number_to_nuscenes_timestamp(self, input_timestamp):
        # Ensure the input is a string representing a 6-digit number
        if len(input_timestamp) != 6 or not input_timestamp.isdigit():
            raise ValueError("Input should be a 6-digit string, e.g., '000620'.")

        # Calculate the total time in seconds (each digit represents 0.1 seconds)
        total_seconds = int(input_timestamp) * 0.1  # Convert to seconds

        # Define the start time (2025-01-01 09:00:00) as the base reference
        start_time = datetime(2025, 1, 1, 9, 0, 0)

        # Add the total seconds to the start time
        target_time = start_time + timedelta(seconds=total_seconds)

        # Convert to microseconds since Unix epoch
        epoch = datetime(1970, 1, 1)
        microseconds_since_epoch = int((target_time - epoch).total_seconds() * 1e6)

        return microseconds_since_epoch

    def _check_instance_annotations(self, instances_chunk, sample_annotations):
        """Helper function for parallel annotation verification"""
        bad_instances = []
        all_ann_tokens = {ann['token'] for ann in sample_annotations}

        for instance in instances_chunk:
            unlinked = []
            for ann_token in instance['annonation_tokens']:
                if ann_token not in all_ann_tokens:
                    unlinked.append(ann_token)
            if unlinked:
                bad_instances.append(instance)
        return bad_instances

    def _verify_metadata_integrity(self):
        """Verify the integrity of generated metadata"""
        print("Verifying metadata integrity...")

        # Check sample and scene links
        for scene in self.metadata['scene']:
            assert (
                scene['nbr_samples'] >= self.MIN_SCENE_LENGTH
            ), f"Scene {scene['token']} has less than {self.MIN_SCENE_LENGTH} samples"

            first_token = scene['first_sample_token']
            last_token = scene['last_sample_token']

            prev_sample_token = None
            next_sample_token = None
            for sample in self.metadata['sample']:
                if prev_sample_token is not None:
                    assert (
                        sample['prev'] == prev_sample_token
                    ), f"Invalid prev link in scene {scene['token']}, sample {sample['token']}, expected {prev_sample_token}, got {sample['prev']}"
                    prev_sample_token = sample['token']

                if next_sample_token is not None:
                    assert (
                        sample['token'] == next_sample_token
                    ), f"Invalid next link in scene {scene['token']}, sample {sample['token']}"
                    next_sample_token = sample['next']

                if sample['token'] == first_token:
                    prev_sample_token = sample['token']
                    next_sample_token = sample['next']

                if sample['token'] == last_token:
                    break

        # Parallelize instance-annotation checks
        print("Verifying instance-annotation links with multiprocessing...")
        instances = self.metadata['instance']
        sample_anns = self.metadata['sample_annotation']
        chunk_size = max(1, len(instances) // (self.num_workers * 4))
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            process_chunk = partial(
                self._check_instance_annotations, sample_annotations=sample_anns
            )
            chunks = (
                instances[i : i + chunk_size]
                for i in range(0, len(instances), chunk_size)
            )

            bad_instances = []
            for result in pool.imap_unordered(process_chunk, chunks, chunksize=4):
                bad_instances.extend(result)

        # Remove bad instances
        if bad_instances:
            print(
                f"Removing {len(bad_instances)} instances with unlinked annotations..."
            )
            self.metadata['instance'] = [
                inst for inst in self.metadata['instance'] if inst not in bad_instances
            ]

        print("Metadata integrity verification completed.")

    def save_metadata(self):
        """Save all metadata files"""
        for name, data in self.metadata.items():
            output_path = os.path.join(self.target_dir, self.version, f'{name}.json')
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

    def convert(self, invalid_frames: dict = None):
        """Main conversion process"""
        print("Creating directory structure...")
        self.create_directory_structure()

        print("Copying sensor data...")
        self.copy_sensor_data()

        print("Loading source meta data...")
        self.load_metadata()

        if self.side == "cooperative":
            print("Checking invalid frames...")
            invalid_frames = self.check_invalid_frames()
        else:
            assert (
                invalid_frames is not None
            ), "Invalid frames should be provided for non-cooperative side"
        print(f"Removing {len(invalid_frames)} invalid frames...")
        self.remove_invalid_frames(invalid_frames)

        print("Generating metadata...")
        self.generate_metadata()

        print("Saving metadata...")
        self.save_metadata()

        print("Conversion completed!")

        if self.side == "cooperative":
            return invalid_frames

    def verify_split_scenes(self, split_file: str):
        """Verify split scenes"""
        print(f"Verifying split scenes for {self.side} side...")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        train_scene_names = split_data['batch_split']['train']
        val_scene_names = split_data['batch_split']['val']

        available_scene_names = [scene['name'] for scene in self.metadata['scene']]

        for scene_name in train_scene_names:
            assert (
                scene_name in available_scene_names
            ), f"Scene {scene_name} from split file not found in metadata"
        for scene_name in val_scene_names:
            assert (
                scene_name in available_scene_names
            ), f"Scene {scene_name} from split file not found in metadata"

    def complete_early_fusion_sensors(self):
        """Complete early fusion sensors. Only need to load cam_air data, other sensor calibs, poses and annotations are already loaded"""
        self.sensor_types = {
            "front": "cam",
            "back": "cam",
            "left": "cam",
            "right": "cam",
            "top": "lidar",
            "front_air": "cam_air",
            "back_air": "cam_air",
            "left_air": "cam_air",
            "right_air": "cam_air",
            "bottom_air": "cam_air",
            "top_air": "lidar_air",
        }

        for sensor_name, sensor_type in self.sensor_types.items():
            if sensor_type == 'cam_air':
                sensor_name_air = sensor_name.split('_')[0]
                self.calibration_data[sensor_name] = load_calibration(
                    self.source_dir_air, sensor_name_air
                )
                self.metadata['sensor'].append(
                    {
                        'token': self._generate_token(f'sensor_{sensor_name}'),
                        'channel': f'{sensor_type.split("_")[0]}_{sensor_name}'.upper(),
                        'modality': sensor_type.split('_')[0].upper(),
                    }
                )

    def _complete_early_fusion_calibrated_sensor_metadata(self):
        """Complete early fusion calibrated sensor metadata. Air sensor calibs are different for each frame"""
        T_top_lidar_to_ego = self.calibration_data['top'].extrinsic

        for sensor_name, calib_data in self.calibration_data.items():
            if 'air' not in sensor_name:
                continue

            for idx, frame in enumerate(self.frame_list):
                timestamp = self.timestamp_list[idx]

                T_air_sensor_to_air = calib_data.extrinsic
                T_air_to_ENU, _ = trans_ego_vs_ENU(self.ego_poses_air[frame])
                _, T_ENU_TO_ego = trans_ego_vs_ENU(self.ego_poses[frame])

                T_air_sensor_to_ego = T_ENU_TO_ego @ T_air_to_ENU @ T_air_sensor_to_air
                T_air_sensor_to_lidar = (
                    np.linalg.inv(T_top_lidar_to_ego) @ T_air_sensor_to_ego
                )
                T_lidar_to_air_sensor = np.linalg.inv(T_air_sensor_to_lidar)

                self.metadata['calibrated_sensor'].append(
                    {
                        'token': self._generate_token(
                            f'calibrated_sensor_{sensor_name}_{timestamp}'
                        ),
                        'sensor_token': self._get_sensor_token(sensor_name),
                        'translation': T_air_sensor_to_ego[:3, 3].tolist(),
                        'rotation': Quaternion(
                            matrix=T_air_sensor_to_ego[:3, :3]
                        ).elements.tolist(),
                        'camera_intrinsic': (
                            calib_data.intrinsic.tolist()
                            if sensor_name != 'top' and sensor_name != 'top_air'
                            else []
                        ),
                        'camera_distortion': [],  # Placeholder
                        'lidar2cam_translation': T_lidar_to_air_sensor[:3, 3].tolist(),
                        'lidar2cam_rotation': Quaternion(
                            matrix=T_lidar_to_air_sensor[:3, :3]
                        ).elements.tolist(),
                        'sensor2lidar_translation': T_air_sensor_to_lidar[
                            :3, 3
                        ].tolist(),
                        'sensor2lidar_rotation': Quaternion(
                            matrix=T_air_sensor_to_lidar[:3, :3]
                        ).elements.tolist(),
                    }
                )

    def _complete_early_fusion_sample_data_metadata(self):
        """Complete early fusion sample data metadata. Only need to generate cam_air sample data"""
        for sensor_name, sensor_type in self.sensor_types.items():
            if sensor_type != 'cam_air':
                continue

            prev_token = ''
            scene_num = 0
            for idx, frame in enumerate(self.frame_list):
                timestamp = self.timestamp_list[idx]

                is_next_sample_continuous = (idx < len(self.frame_list) - 1) and (
                    int(self.frame_list[idx + 1]) - int(frame) == 1
                )
                # is_next_sample_continuous = True

                next_token = (
                    self._generate_token(
                        f'data_{sensor_name}_{self.timestamp_list[idx + 1]}'
                    )
                    if is_next_sample_continuous
                    else ''
                )

                sample_data = {
                    'token': self._generate_token(f'data_{sensor_name}_{timestamp}'),
                    'sample_token': self._generate_token(f'sample_{timestamp}'),
                    'ego_pose_token': (self._generate_token(f'ego_pose_{timestamp}')),
                    'calibrated_sensor_token': self._generate_token(
                        f'calibrated_sensor_{sensor_name}_{timestamp}'
                    ),
                    'timestamp': timestamp,
                    'prev': prev_token,
                    'next': next_token,
                    'is_key_frame': True,
                    'filename': os.path.join(
                        'samples',
                        f"{sensor_type.split('_')[0]}_{sensor_name}".upper(),
                        f"{frame}.png",
                    ),
                    'fileformat': 'png',
                    'width': 1920,
                    'height': 1080,
                }

                self.metadata['sample_data'].append(sample_data)

                if is_next_sample_continuous:
                    prev_token = sample_data['token']
                else:
                    prev_token = ''
                    scene_num += 1

    def convert_early_fusion(self):
        """Convert early fusion data"""
        print("Converting early fusion data...")

        # self.target_dir = os.path.join(self.target_dir, os.pardir, "early-fusion")
        self.target_dir = self.target_dir.parent / "early-fusion"
        self.side = "early-fusion"

        print("Creating directory structure...")
        self.create_directory_structure()

        print("Completing early fusion sensors...")
        self.complete_early_fusion_sensors()

        print("Copying sensor data...")
        self.copy_sensor_data()

        print("Completing early fusion metadata...")
        self._complete_early_fusion_calibrated_sensor_metadata()
        self._complete_early_fusion_sample_data_metadata()

        print("Saving metadata...")
        self.save_metadata()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Griffin data to NuScenes format"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="datasets/griffin_50scenes_25m/griffin-release",
        help="Path to the source data directory",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="datasets/griffin_50scenes_25m/griffin-nuscenes",
        help="Path to the target data directory",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="data/split_datas/griffin_50scenes_25m.json",
        help="Path to the split data json file",
    )
    parser.add_argument(
        "--early_fusion",
        default=False,
        action="store_true",
        help="Convert early fusion data",
    )
    args = parser.parse_args()

    print("Converting cooperative side data...")
    cooperative_converter = GriffinKittiToNuScenesConverter(
        source_dir=os.path.join(args.source_dir),
        target_dir=os.path.join(args.target_dir, "cooperative"),
        side="cooperative",
    )
    invalid_frames = cooperative_converter.convert()
    if args.early_fusion:
        cooperative_converter.convert_early_fusion()

    print("Converting drone side data...")
    drone_converter = GriffinKittiToNuScenesConverter(
        source_dir=os.path.join(args.source_dir, "drone-side"),
        target_dir=os.path.join(args.target_dir, "drone-side"),
        side="drone",
    )
    drone_converter.convert(invalid_frames)

    print("Converting vehicle side data...")
    vehicle_converter = GriffinKittiToNuScenesConverter(
        source_dir=os.path.join(args.source_dir, "vehicle-side"),
        target_dir=os.path.join(args.target_dir, "vehicle-side"),
        side="vehicle",
    )
    vehicle_converter.convert(invalid_frames)

    cooperative_converter.verify_split_scenes(args.split_file)
    drone_converter.verify_split_scenes(args.split_file)
    vehicle_converter.verify_split_scenes(args.split_file)

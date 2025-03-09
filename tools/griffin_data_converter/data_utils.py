from dataclasses import dataclass
from typing import List
import numpy as np
import cv2
import os
import json


def ensure_angles_in_degrees(angles: List[float]) -> None:
    """
    Ensure angles are in degrees, rather than radians
    """
    radian_threshold = 2 * np.pi
    warn_threshold = np.pi

    has_large_value = any(abs(a) > radian_threshold for a in angles)
    if has_large_value:
        return

    has_medium_value = any(abs(a) > warn_threshold for a in angles)
    if has_medium_value:
        print("Warning: Input angles may be in radians, please check.")


@dataclass
class CalibrationParams:
    """
    Stores camera calibration parameters.
    extrinsic: Matrix to project points from camera to Ego coordinate system, shape (4, 4).
    intrinsic: Matrix to project points from camera to pixel coordinate system, shape (3, 3).
    """

    extrinsic: np.ndarray
    intrinsic: np.ndarray


@dataclass
class PoseData:
    """
    Stores Ego pose data in ENU coordinate system, xyz in meters, RPY euler angles in degrees.
    """

    x: float = 0.0  # East
    y: float = 0.0  # North
    z: float = 0.0  # Height from ground
    roll: float = 0.0  # + indicates left side up
    pitch: float = 0.0  # + indicates nose down
    yaw: float = 0.0  # 0 indicates East, + indicates counterclockwise

    # def __post_init__(self):
    #     ensure_angles_in_degrees([self.roll, self.pitch, self.yaw])


@dataclass
class ObjectData:
    """
    Stores object data in Ego coordinates, xyzlwh in meters, RPY euler angles in degrees, visibility in [0, 1].
    """

    type: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    l: float = 0.0
    w: float = 0.0
    h: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    id: int = 0
    visibility: float = 1.0

    def __post_init__(self):
        """
        Convert object type to standard format, especially for string input.
        """
        # Convert object type to standard format
        if self.type in ["Alfa", "Charlie", "Delta"]:
            self.type = "Soldier"
        elif self.type in [
            "ZTZ99A",
            "HMARS",
            "M1A2SEP",
            "M109",
            "M2A3",
            "T72B3",
            "ZTZ96A",
            "ZTL11",
            "PGZ09",
            "2s3m",
            "T90A",
            "Foxtrot",
            "Mar1a3",
        ]:
            self.type = "Military"
        # elif self.type in ["Car", "Truck", "Bus"]:
        #     self.type = "Vehicle"

        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.l = float(self.l)
        self.w = float(self.w)
        self.h = float(self.h)

        self.roll = float(self.roll)
        self.pitch = float(self.pitch)
        self.yaw = float(self.yaw)
        # ensure_angles_in_degrees([self.roll, self.pitch, self.yaw])

        self.id = int(self.id)
        self.visibility = float(self.visibility)

    def __str__(self):
        return f'Object type: {self.type}, position: ({self.x}, {self.y}, {self.z}), dimensions: ({self.l}, {self.w}, {self.h}), RPY rotation: ({self.roll}, {self.pitch}, {self.yaw}), id: {self.id}, visibility: {self.visibility}'


def load_scene_infos(data_base_path: str):
    """
    Load scene data for the whole dataset.
    """
    scene_file = os.path.join(data_base_path, "scene_infos.json")
    with open(scene_file, 'r') as f:
        data = json.load(f)
    frame2scene = {}
    for idx, scene_info in enumerate(data):
        for frame in scene_info['info']['frames']:
            frame2scene[frame] = idx
    return data, frame2scene


def load_label(data_base_path: str, frame: str):
    """
    Load label data for a given frame.
    Args:
        data_base_path (str): Base path to data directory
        frame (str): Frame number
    Returns:
        List of ObjectData instances
    """
    label_file = os.path.join(data_base_path, "label", f'{frame}.txt')

    if not os.path.exists(label_file):
        raise FileNotFoundError(f'Label file {label_file} not found')

    with open(label_file, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        print(f"Warning: label file {label_file} is empty")
        return []

    expected_attributes = [
        'type',
        'x',
        'y',
        'z',
        'l',
        'w',
        'h',
        'roll',
        'pitch',
        'yaw',
        'id',
        'visibility',
    ]
    attribute_num = len(lines[0].strip().split())

    objects = []
    if attribute_num == len(expected_attributes):
        for line in lines:
            objects.append(ObjectData(*line.strip().split()))
    elif attribute_num == len(expected_attributes) - 1:
        print("Warning: missing visibility")
        for line in lines:
            objects.append(ObjectData(*line.strip().split(), visibility=1.0))
    elif attribute_num == len(expected_attributes) - 3:
        print("Warning: missing roll, pitch, visibility")
        for line in lines:
            objects.append(
                ObjectData(
                    type=line[0],
                    x=line[1],
                    y=line[2],
                    z=line[3],
                    l=line[4],
                    w=line[5],
                    h=line[6],
                    roll=0.0,
                    pitch=0.0,
                    yaw=line[7],
                    id=line[8],
                    visibility=1.0,
                )
            )
    else:
        raise ValueError(f'Expected 9 or 12 attributes, but got {attribute_num}')

    return objects


def load_calibration(data_base_path: str, sensor: str) -> CalibrationParams:
    """
    Load calibration parameters for a given sensor.
    Args:
        data_base_path (str): Base path to data directory
        sensor (str): Sensor name
    Returns:
        CalibrationParams: Camera calibration parameters
    """
    calib_file = os.path.join(data_base_path, "calib", f'{sensor}.json')

    if not os.path.exists(calib_file):
        raise FileNotFoundError(f'Calibration file {calib_file} not found')

    with open(calib_file, 'r') as f:
        data = json.load(f)

    return CalibrationParams(
        extrinsic=np.array(data['extrinsic']),
        intrinsic=np.array(data.get('intrinsic', None)),
    )


def load_pose(data_base_path: str, frame: str) -> PoseData:
    """
    Load pose data for a given frame.
    Args:
        data_base_path (str): Base path to data directory
        frame (str): Frame number
    Returns:
        PoseData: Pose data
    """
    pose_file = os.path.join(data_base_path, "pose", f'{frame}.json')

    if not os.path.exists(pose_file):
        raise FileNotFoundError(f'Pose data file {pose_file} not found')

    with open(pose_file, 'r') as f:
        data = json.load(f)
    return PoseData(
        x=data['x'],
        y=data['y'],
        z=data['z'],
        pitch=data['pitch'],
        roll=data['roll'],
        yaw=data['yaw'],
    )


def load_img(data_base_path: str, direction: str, frame: str):
    """
    Load camera image for a given frame.
    Args:
        data_base_path (str): Base path to data directory
        direction (str): Camera direction
        frame (str): Frame number
    Returns:
        np.ndarray: Camera image, shape (H, W, 3), dtype uint8, BGR format
    """
    img_file = os.path.join(data_base_path, "camera", direction, f'{frame}.png')

    if not os.path.exists(img_file):
        raise FileNotFoundError(f'Camera image file {img_file} not found')

    return cv2.imread(img_file)

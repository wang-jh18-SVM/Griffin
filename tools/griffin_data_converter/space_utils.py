import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from data_utils import CalibrationParams, PoseData, ObjectData


def trans_sensor_vs_ego(sensor_extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transform matrices between Sensor and Ego coordinate systems.
    Args:
        sensor_extrinsic (np.ndarray): Sensor extrinsic matrix, shape (4, 4)
    Returns:
        Tuple[np.ndarray, np.ndarray]: transform matrices T_sensor_to_ego, T_ego_to_sensor, both shape (4, 4)
    """
    T_sensor_to_ego = sensor_extrinsic
    T_ego_to_sensor = np.linalg.inv(sensor_extrinsic)
    return T_sensor_to_ego, T_ego_to_sensor


def trans_ego_vs_ENU(pose_data: PoseData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transform matrices between Ego and ENU coordinate systems.
    Args:
        pose_data (PoseData): Pose data
    Returns:
        Tuple[np.ndarray, np.ndarray]: transform matrices T_ego_to_ENU, T_ENU_to_ego, both shape (4, 4)
    """
    R_ego_to_ENU = R.from_euler(
        'xyz', [pose_data.roll, pose_data.pitch, pose_data.yaw], degrees=True
    ).as_matrix()
    t_ego_to_ENU = np.array([pose_data.x, pose_data.y, pose_data.z])
    T_ego_to_ENU = np.eye(4)
    T_ego_to_ENU[:3, :3] = R_ego_to_ENU
    T_ego_to_ENU[:3, 3] = t_ego_to_ENU

    T_ENU_to_ego = np.linalg.inv(T_ego_to_ENU)
    return T_ego_to_ENU, T_ENU_to_ego


def trans_object_vs_ego(obj: ObjectData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transform matrices between Object and Ego coordinate systems.
    Args:
        obj (ObjectData): Object data
    Returns:
        Tuple[np.ndarray, np.ndarray]: transform matrices T_obj_to_ego, T_ego_to_obj, both shape (4, 4)
    """
    R_obj_to_ego = R.from_euler(
        'xyz', [obj.roll, obj.pitch, obj.yaw], degrees=True
    ).as_matrix()
    t_obj_to_ego = np.array([obj.x, obj.y, obj.z])

    T_obj_to_ego = np.eye(4)
    T_obj_to_ego[:3, :3] = R_obj_to_ego
    T_obj_to_ego[:3, 3] = t_obj_to_ego

    T_ego_to_obj = np.linalg.inv(T_obj_to_ego)
    return T_obj_to_ego, T_ego_to_obj


def proj_object_to_ego(
    pts_obj: np.ndarray,
    obj: ObjectData,
) -> np.ndarray:
    """
    Project object to Ego coordinate system.
    Args:
        pts_obj (np.ndarray): Points in object coordinate system, shape (N, 3)
        obj (ObjectData): Object data
    Returns:
        np.ndarray: Projected points in Ego coordinates, shape (N, 3)
    """
    T_obj_to_ego, _ = trans_object_vs_ego(obj)

    pts_obj_homo = np.hstack((pts_obj, np.ones((pts_obj.shape[0], 1))))
    pts_ego_homo = (T_obj_to_ego @ pts_obj_homo.T).T

    return pts_ego_homo[:, :3]


def proj_sensor_to_ego(
    pts_sensor: np.ndarray,
    calib_param: CalibrationParams,
) -> np.ndarray:
    """
    Project points from Sensor to Ego coordinate system.
    Args:
        pts_sensor (np.ndarray): Points in Sensor coordinate system, shape (N, 3)
        calib_param (CalibrationParams): Camera calibration parameters
    Returns:
        np.ndarray: Projected points in Ego coordinates, shape (N, 3)
    """
    T_sensor_to_ego, _ = trans_sensor_vs_ego(calib_param.extrinsic)

    pts_sensor_homo = np.hstack((pts_sensor, np.ones((pts_sensor.shape[0], 1))))
    pts_ego_homo = (T_sensor_to_ego @ pts_sensor_homo.T).T

    return pts_ego_homo[:, :3]


def proj_ego_to_sensor(
    pts_ego: np.ndarray,
    calib_param: CalibrationParams,
) -> np.ndarray:
    """
    Project points from Ego to Sensor coordinate system.
    Args:
        pts_ego (np.ndarray): Points in Ego coordinate system, shape (N, 3)
        calib_param (CalibrationParams): Camera calibration parameters
    Returns:
        np.ndarray: Projected points in Sensor coordinates, shape (N, 3)
    """
    _, T_ego_to_sensor = trans_sensor_vs_ego(calib_param.extrinsic)

    pts_ego_homo = np.hstack((pts_ego, np.ones((pts_ego.shape[0], 1))))
    pts_sensor_homo = (T_ego_to_sensor @ pts_ego_homo.T).T

    return pts_sensor_homo[:, :3]


def proj_ego_to_ENU(
    pts_ego: np.ndarray,
    pose_data: PoseData,
) -> np.ndarray:
    """
    Project points from Ego to ENU coordinate system.
    Args:
        pts_ego (np.ndarray): Points in Ego coordinate system, shape (N, 3)
        pose_data (PoseData): Pose data
    Returns:
        np.ndarray: Projected points in ENU coordinates, shape (N, 3)
    """
    T_ego_to_ENU, _ = trans_ego_vs_ENU(pose_data)

    pts_ego_homo = np.hstack((pts_ego, np.ones((pts_ego.shape[0], 1))))
    pts_ENU_homo = (T_ego_to_ENU @ pts_ego_homo.T).T

    return pts_ENU_homo[:, :3]


def proj_ENU_to_ego(
    pts_ENU: np.ndarray,
    pose_data: PoseData,
) -> np.ndarray:
    """
    Project points from ENU to Ego coordinate system.
    Args:
        pts_ENU (np.ndarray): Points in ENU coordinate system, shape (N, 3)
        pose_data (PoseData): Pose data
    Returns:
        np.ndarray: Projected points in Ego coordinates, shape (N, 3)
    """
    _, T_ENU_to_ego = trans_ego_vs_ENU(pose_data)

    pts_ENU_homo = np.hstack((pts_ENU, np.ones((pts_ENU.shape[0], 1))))
    pts_ego_homo = (T_ENU_to_ego @ pts_ENU_homo.T).T

    return pts_ego_homo[:, :3]


def proj_sensor_to_ENU(
    pts_sensor: np.ndarray,
    calib_param: CalibrationParams,
    pose_data: PoseData,
) -> np.ndarray:
    """
    Project points from camera to ENU coordinate system.
    Args:
        pts_cam (np.ndarray): Points in camera coordinate system, shape (N, 3)
        calib_param (CalibrationParams): Camera calibration parameters
        pose_data (PoseData): Pose data
    Returns:
        np.ndarray: Projected points in ENU coordinates, shape (N, 3)
    """
    T_sensor_to_ego, _ = trans_sensor_vs_ego(calib_param.extrinsic)
    T_ego_to_ENU, _ = trans_ego_vs_ENU(pose_data)

    T_sensor_to_ENU = T_ego_to_ENU @ T_sensor_to_ego

    pts_sensor_homo = np.hstack((pts_sensor, np.ones((pts_sensor.shape[0], 1))))
    pts_ENU_homo = (T_sensor_to_ENU @ pts_sensor_homo.T).T

    return pts_ENU_homo[:, :3]


def proj_ENU_to_sensor(
    pts_ENU: np.ndarray,
    calib_param: CalibrationParams,
    pose_data: PoseData,
) -> np.ndarray:
    """
    Project points from ENU to camera coordinate system.
    Args:
        pts_ENU (np.ndarray): Points in ENU coordinate system, shape (N, 3)
        calib_param (CalibrationParams): Camera calibration parameters
        pose_data (PoseData): Pose data
    Returns:
        np.ndarray: Projected points in camera coordinates, shape (N, 3)
    """
    _, T_ego_to_sensor = trans_sensor_vs_ego(calib_param.extrinsic)
    _, T_ENU_to_ego = trans_ego_vs_ENU(pose_data)

    T_ENU_to_sensor = T_ego_to_sensor @ T_ENU_to_ego

    pts_ENU_homo = np.hstack((pts_ENU, np.ones((pts_ENU.shape[0], 1))))
    pts_sensor_homo = (T_ENU_to_sensor @ pts_ENU_homo.T).T

    return pts_sensor_homo[:, :3]


def proj_cam_to_img(
    pts_cam: np.ndarray,
    calib_param: CalibrationParams,
    img_shape: Tuple[int, int] = [1920, 1080],
    return_valid_index: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project points from the camera coordinate system to the image plane.
    Args:
        pts_cam (np.ndarray): Points in the camera coordinate system, shape (N, 3).
        calib_param (CalibrationParams): Camera calibration parameters
        img_shape (Tuple[int, int]): Shape of the image (width, height).
        return_valid_index (bool): Whether to return valid indices.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - pts_img (np.ndarray): A (M, 2) array of 2D points in the image plane.
            - depth (np.ndarray): A (M, 1) array of depth values for the points.
            - valid_index (np.ndarray): A (N, 1) array of boolean values indicating whether the point is valid.
    """
    depth = pts_cam[:, 2][:, np.newaxis]
    pts_img = (calib_param.intrinsic @ pts_cam.T).T
    pts_img = pts_img / depth
    pts_img = np.round(pts_img).astype(int)

    valid_index = (
        (pts_img[:, 0] >= 0)
        & (pts_img[:, 0] < img_shape[0])
        & (pts_img[:, 1] >= 0)
        & (pts_img[:, 1] < img_shape[1])
        & (depth[:, 0] > 0)
    )

    if not return_valid_index:
        return (
            pts_img[valid_index],
            depth[valid_index],
        )
    else:
        return (
            pts_img[valid_index],
            depth[valid_index],
            [i for i, val in enumerate(valid_index) if val],
        )


def get_uav_ideal_depth(
    pixels: np.ndarray,
    calib_param: CalibrationParams,
    pose_data: PoseData,
    max_depth: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ideal depth of UAV pixels from the camera to the ground plane.
    Args:
        pixels (np.ndarray): Array of pixel coordinates in the image.
        calib_param (CalibrationParams): Calibration parameters of the camera.
        pose_data (PoseData): Pose data containing the UAV's position and orientation.
        max_depth (float): Max depth prediction, default to 300m.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - depth (np.ndarray): Depth values for each pixel.
            - pts_ground_ENU (np.ndarray): Intersection points of rays with the ground plane in ENU coordinates.
            - cam_pos_ENU (np.ndarray): Camera position in ENU coordinates.
            - pts_ground_ego (np.ndarray): Intersection points of rays with the ground plane in Ego coordinates.
    """
    # Camera position in ENU coordinates
    cam_pos_cam = np.array([[0, 0, 0]])
    cam_pos_ENU = proj_sensor_to_ENU(cam_pos_cam, calib_param, pose_data)[0]

    # Add depth of 1 to pixels
    pixels_homo = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
    # Get pixels in camera coordinates
    pixels_cam = (np.linalg.inv(calib_param.intrinsic) @ pixels_homo.T).T
    # Normalize pixels to get unit length
    pixels_cam = pixels_cam / np.linalg.norm(pixels_cam, axis=1)[:, np.newaxis]

    # Get unit vectors in ENU coordinates
    pixels_ENU = proj_sensor_to_ENU(pixels_cam, calib_param, pose_data)
    rays_ENU = pixels_ENU - cam_pos_ENU
    rays_ENU[:, 2] = np.where(rays_ENU[:, 2] < 0, rays_ENU[:, 2], np.nan)

    # Calculate depth for intersection with ground plane z=0
    depth = (0 - cam_pos_ENU[2]) / rays_ENU[:, 2]
    depth = np.where(depth <= max_depth, depth, np.nan)

    # Calculate intersection points
    pts_ground_ENU = cam_pos_ENU + rays_ENU * depth[:, np.newaxis]

    # Convert to Ego coordinates
    pts_ground_ego = proj_ENU_to_ego(pts_ground_ENU, pose_data)

    return depth, pts_ground_ENU, cam_pos_ENU, pts_ground_ego

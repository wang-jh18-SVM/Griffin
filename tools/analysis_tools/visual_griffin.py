import mmcv
import argparse
import os
import glob
import cv2
from nuscenes.nuscenes import NuScenes
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import (
    Box,
)
from nuscenes.utils.geometry_utils import (
    box_in_image,
    BoxVisibility,
)
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.utils import boxes_to_sensor

# Initialize TRACKING_NAMES with Griffin dataset class names to fix the assertion error
# This is necessary because the TRACKING_NAMES list is empty by default
from nuscenes.eval.tracking.data_classes import TRACKING_NAMES

# Define valid tracking names for Griffin dataset
VALID_TRACKING_NAMES = [
    'car',
    'bicycle',
    'pedestrian',
]

# Set TRACKING_NAMES globally to allow TrackingBox to validate correctly
TRACKING_NAMES[:] = VALID_TRACKING_NAMES


# Define mapping from category names to tracking names
def get_tracking_name(category_name: str) -> str:
    """
    Convert category name to valid tracking name format for Griffin dataset.

    For Griffin dataset, the tracking names are the simple category names
    like 'car', 'pedestrian', etc., not the NuScenes format names.

    :param category_name: Original category name
    :return: Valid tracking name
    """
    # If it contains a dot (like 'vehicle.car'), convert to simple name
    if '.' in category_name:
        # Extract the part after the last dot
        simple_name = category_name.split('.')[-1]

        # Special case mappings
        special_cases = {
            'bus.rigid': 'bus',
            'bus.bendy': 'bus',
            'truck.rigid': 'truck',
            'pedestrian.adult': 'pedestrian',
            'pedestrian.child': 'pedestrian',
            'pedestrian.construction_worker': 'pedestrian',
            'pedestrian.police_officer': 'pedestrian',
        }

        # Check special cases first
        for case, mapping in special_cases.items():
            if category_name.endswith(case):
                return mapping

        # Otherwise return the simple name if valid
        if simple_name in VALID_TRACKING_NAMES:
            return simple_name

    # If it's already a simple name, check if it's valid
    if category_name in VALID_TRACKING_NAMES:
        return category_name

    # Default to 'car' if not recognized (this is a fallback)
    print(f"Warning: Unknown category '{category_name}', defaulting to 'car'")
    return 'car'


# Colorful bounding box colors for different tracking IDs
color_mapping = [
    np.array([1.0, 0.0, 0.0]),  # Red
    np.array([1.0, 0.078, 0.576]),  # Pink
    np.array([0.0, 0.0, 1.0]),  # Blue
    np.array([1.0, 1.0, 0.0]),  # Yellow
    np.array([1.0, 0.647, 0.0]),  # Orange
    np.array([0.502, 0.0, 0.502]),  # Purple
    np.array([0.0, 1.0, 1.0]),  # Cyan
    np.array([1.0, 0.0, 1.0]),  # Magenta
    np.array([0.0, 1.0, 0.502]),  # Teal
    np.array([1.0, 0.843, 0.0]),  # Gold
]

# Define distinct colors for different classes and sources (GT vs prediction)
class_colors = {
    'car': {
        'gt': np.array([0.0, 0.5, 0.0]),  # Dark green for GT cars
        'pred': np.array(
            [0.2, 0.9, 0.5, 0.8]
        ),  # Mint green with opacity for predicted cars
    },
    'bicycle': {
        'gt': np.array([0.0, 0.0, 0.7]),  # Dark blue for GT bicycles
        'pred': np.array(
            [0.5, 0.7, 1.0, 0.8]
        ),  # Light blue with opacity for predicted bicycles
    },
    'pedestrian': {
        'gt': np.array([0.7, 0.0, 0.0]),  # Dark red for GT pedestrians
        'pred': np.array(
            [1.0, 0.6, 0.0, 0.8]
        ),  # Orange with opacity for predicted pedestrians
    },
    'default': {
        'gt': np.array([0.4, 0.4, 0.4]),  # Dark gray for GT other objects
        'pred': np.array(
            [0.9, 0.9, 0.2, 0.8]
        ),  # Yellow with opacity for predicted other objects
    },
}


def get_box_color(box_name, is_gt=True):
    """
    Get color for a box based on its class and whether it's ground truth or prediction

    :param box_name: Name/class of the box (e.g., 'car', 'bicycle', 'pedestrian')
    :param is_gt: Whether the box is ground truth (True) or prediction (False)
    :return: RGB color array
    """
    # Determine the source (gt or pred)
    source = 'gt' if is_gt else 'pred'

    # Convert box name to lowercase for consistent matching
    name_lower = box_name.lower()

    # Match to known classes
    if 'car' in name_lower:
        return class_colors['car'][source]
    elif 'bicycle' in name_lower:
        return class_colors['bicycle'][source]
    elif 'pedestrian' in name_lower:
        return class_colors['pedestrian'][source]
    else:
        return class_colors['default'][source]


def visualize_sample(
    nusc: NuScenes,
    sample_token: str,
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    conf_th: float = 0.15,
    eval_range: list = [-70.0, -40.0, 70.0, 40.0],
    savepath: str = None,
    ax=None,
    style: str = 'bev',
    x_center: float = 0.0,
    y_center: float = 0.0,
    gt_linewidth: float = 1.0,
    pred_linewidth: float = 2.0,
) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param conf_th: The confidence threshold used to filter detections.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param savepath: If given, saves the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    :param style: 'bev' or 'utm'.
    :param x_center: The x center of the plot.
    :param y_center: The y center of the plot.
    :param gt_linewidth: Line width for ground truth bounding boxes.
    :param pred_linewidth: Line width for prediction bounding boxes.
    """
    # Retrieve sensor & pose records
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    if style == 'bev':
        # Map GT boxes to lidar
        boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)
        # Map EST boxes to lidar
        boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)
    elif style == 'utm':
        # Map GT boxes to global
        boxes_gt = []
        for box in boxes_gt_global:
            box = Box(box.translation, box.size, Quaternion(box.rotation))
            boxes_gt.append(box)

        # Map EST boxes to global
        boxes_est = []
        for box in boxes_est_global:
            box = Box(box.translation, box.size, Quaternion(box.rotation))
            boxes_est.append(box)

    # Add scores to EST boxes
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.tracking_score
        box_est.tracking_id = box_est_global.tracking_id
        box_est.name = (
            box_est_global.tracking_name
            if hasattr(box_est_global, 'tracking_name')
            else ''
        )

    # Init axes
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show ego vehicle
    if style == 'bev':
        ax.plot(0, 0, 'x', color='black')
    elif style == 'utm':
        ego_position = pose_record['translation']
        ax.plot(ego_position[0], ego_position[1], 'x', color='black')

    # Show GT boxes with class-specific colors
    for box in boxes_gt:
        if hasattr(box, 'tracking_name'):
            c = get_box_color(box.tracking_name, is_gt=True)
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=gt_linewidth)
        else:
            box.render(
                ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=gt_linewidth
            )

    # Show EST boxes with class-specific colors
    for box in boxes_est:
        # Show only predictions with a high score
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            # Use class-specific colors for predictions
            if hasattr(box, 'name') and box.name:
                c = get_box_color(box.name, is_gt=False)
                box.render(
                    ax, view=np.eye(4), colors=(c, c, c), linewidth=pred_linewidth
                )
            else:
                # Fallback to tracking ID based coloring if name not available
                tr_id = box.tracking_id if hasattr(box, 'tracking_id') else 0
                c = color_mapping[tr_id % len(color_mapping)]
                box.render(
                    ax, view=np.eye(4), colors=(c, c, c), linewidth=pred_linewidth
                )

    # Limit visible range
    if eval_range:
        ax.set_xlim(eval_range[0], eval_range[2])
        ax.set_ylim(eval_range[1], eval_range[3])
    else:
        axes_limit = 150  # controllable range
        ax.set_xlim(x_center - axes_limit, x_center + axes_limit)
        ax.set_ylim(y_center - axes_limit, y_center + axes_limit)

    # Set axis ticks
    if eval_range and eval_range[0] == -70.0:
        ax.set_xticks([-60, -40, -20, 0, 20, 40, 60])
        ax.set_xticklabels(['-60', '-40', '-20', '0', '20', '40', '60'])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_yticklabels(['-40', '-20', '0', '20', '40'])
    elif eval_range and eval_range[0] == -53.0:
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_xticklabels(['-40', '-20', '0', '20', '40'])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_yticklabels(['-40', '-20', '0', '20', '40'])

    # Set aspect to equal to ensure distance ratios are preserved
    ax.set_aspect('equal')

    # Don't force box aspect to be 1, let it adjust based on the figure size
    # This allows the BEV view to have a width and length more similar to others in (1920,1080)
    # ax.set_box_aspect(1)  # Commented out to allow for different aspect ratios

    if savepath is not None:
        savepath = savepath + ('_bev' if style == 'bev' else '_utm')
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def get_color(category_name: str, nusc: NuScenes):
    """
    Provides the default colors based on the category names.
    """
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def get_sample_data(
    nusc: NuScenes,
    sample_data_token: str,
    sensor_channel: str,
    boxes: list,
    box_vis_level: BoxVisibility = BoxVisibility.ANY,
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.

    :param nusc: NuScenes object
    :param sample_data_token: Sample_data token
    :param sensor_channel: Sensor channel name
    :param boxes: List of boxes to transform
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes
    :return: (data_path, transformed_boxes, camera_intrinsic)
    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])

    # # Get calibrated sensor record for the requested sensor channel
    # sample = nusc.get('sample', sd_record['sample_token'])

    # if sensor_channel not in sample['data']:
    #     return None, [], None

    # cam_sample_data_token = sample['data'][sensor_channel]
    # cam_sd_record = nusc.get('sample_data', cam_sample_data_token)
    # cs_record = nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])

    # Get data path
    # data_path = nusc.get_sample_data_path(cam_sample_data_token)
    data_path = nusc.get_sample_data_path(sample_data_token)

    # Get camera intrinsic matrix if it's a camera
    if sensor_record['modality'] == 'camera' or sensor_record['modality'] == 'CAM':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (1920, 1080)  # Default size for Griffin dataset
    else:
        cam_intrinsic = None
        imsize = None

    # Transform boxes to sensor coordinate system
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box = box.copy()
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        # Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        # Only include boxes that are visible in this sensor
        if (
            sensor_record['modality'] == 'camera' or sensor_record['modality'] == 'CAM'
        ) and not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidar_render(
    nusc: NuScenes,
    sample_token: str,
    pred_data: dict,
    ax1=None,
    ax2=None,
    out_path=None,
    side='vehicle-side',
    thre=0.0,
    x_center=0.0,
    y_center=0.0,
    gt_linewidth=1.0,
    pred_linewidth=2.0,
):
    """
    Render ground truth and predicted boxes in BEV view.

    :param nusc: NuScenes object
    :param sample_token: Sample token
    :param pred_data: Prediction data
    :param ax1: Axes for BEV view
    :param ax2: Axes for UTM view (deprecated, kept for backward compatibility)
    :param out_path: Output path for saving the visualization
    :param side: Agent side ('vehicle-side', 'drone-side')
    :param thre: Detection threshold
    :param x_center: X center for BEV view
    :param y_center: Y center for BEV view
    :param gt_linewidth: Line width for ground truth bounding boxes
    :param pred_linewidth: Line width for prediction bounding boxes
    """
    # Create lists for ground truth and predicted boxes
    bbox_gt_list = []
    bbox_pred_list = []

    # Get ground truth boxes
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        gt_content = nusc.get('sample_annotation', ann)

        # Map category name to tracking name
        tracking_name = get_tracking_name(gt_content['category_name'])

        bbox_gt_list.append(
            TrackingBox(
                sample_token=gt_content['sample_token'],
                translation=tuple(gt_content['translation']),
                size=tuple(gt_content['size']),
                rotation=tuple(gt_content['rotation']),
                velocity=nusc.box_velocity(gt_content['token'])[:2],
                ego_translation=(
                    (0.0, 0.0, 0.0)
                    if 'ego_translation' not in gt_content
                    else tuple(gt_content['ego_translation'])
                ),
                num_pts=(
                    -1 if 'num_pts' not in gt_content else int(gt_content['num_pts'])
                ),
                tracking_name=tracking_name,
                tracking_score=(
                    -1.0
                    if 'tracking_score' not in gt_content
                    else float(gt_content['tracking_score'])
                ),
                tracking_id=gt_content['instance_token'],
            )
        )

    # Get predicted boxes
    if sample_token in pred_data['results']:
        bbox_anns = pred_data['results'][sample_token]
        for pred_content in bbox_anns:
            # Map tracking name to expected format
            tracking_name = get_tracking_name(pred_content['tracking_name'])

            bbox_pred_list.append(
                TrackingBox(
                    sample_token=pred_content['sample_token'],
                    translation=tuple(pred_content['translation']),
                    size=tuple(pred_content['size']),
                    rotation=tuple(pred_content['rotation']),
                    velocity=tuple(pred_content['velocity']),
                    ego_translation=(
                        (0.0, 0.0, 0.0)
                        if 'ego_translation' not in pred_content
                        else tuple(pred_content['ego_translation'])
                    ),
                    num_pts=(
                        -1
                        if 'num_pts' not in pred_content
                        else int(pred_content['num_pts'])
                    ),
                    tracking_name=tracking_name,
                    tracking_score=(
                        -1.0
                        if 'tracking_score' not in pred_content
                        else float(pred_content['tracking_score'])
                    ),
                    tracking_id=pred_content['tracking_id'],
                )
            )

    # Create EvalBoxes objects
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)

    # Set evaluation range based on side
    if side in ['vehicle-side', 'cooperative', 'drone-side']:
        eval_range = [-70.0, -40.0, 70.0, 40.0]
    else:
        raise ValueError('Side Not Found')

    # Visualize in BEV view only
    visualize_sample(
        nusc,
        sample_token,
        gt_annotations,
        pred_annotations,
        conf_th=thre,
        savepath=out_path,
        eval_range=eval_range,
        ax=ax1,
        style='bev',
        x_center=x_center,
        y_center=y_center,
        gt_linewidth=gt_linewidth,
        pred_linewidth=pred_linewidth,
    )

    # UTM view is deprecated and no longer used
    # If ax2 is provided, create an empty plot with a message
    if ax2 is not None:
        ax2.text(
            0.5,
            0.5,
            "UTM view removed",
            horizontalalignment='center',
            verticalalignment='center',
        )
        ax2.axis('off')


def render_single_sensor(
    nusc: NuScenes,
    sample_token: str,
    sensor_channel: str,
    boxes_gt: List[Box],
    boxes_pred: List[Box],
    out_path: str = None,
    figsize=(16, 9),  # Adjusted from (10, 6) to better match 1920x1080 ratio
    dpi=100,
    gt_linewidth: float = 1.0,
    pred_linewidth: float = 2.0,
) -> None:
    """
    Render and save a visualization for a single sensor.

    :param nusc: NuScenes object
    :param sample_token: Sample token
    :param sensor_channel: Sensor channel name
    :param boxes_gt: Ground truth boxes
    :param boxes_pred: Prediction boxes
    :param out_path: Output path for saving the visualization
    :param figsize: Figure size
    :param dpi: DPI for the figure
    :param gt_linewidth: Line width for ground truth bounding boxes
    :param pred_linewidth: Line width for prediction bounding boxes
    """
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data'][sensor_channel]

    # Get data for this sensor
    data_path, boxes_gt_cam, camera_intrinsic = get_sample_data(
        nusc,
        sample_data_token,
        sensor_channel,
        boxes_gt,
    )

    _, boxes_pred_cam, _ = get_sample_data(
        nusc,
        sample_data_token,
        sensor_channel,
        boxes_pred,
    )

    if data_path is None:
        return

    # Create a new figure for this sensor
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Load and display the image
    data = Image.open(data_path)
    ax.imshow(data)

    # Show predicted boxes
    for box in boxes_pred_cam:
        c = get_box_color(box.name, is_gt=False)
        box.render(
            ax,
            view=camera_intrinsic,
            normalize=True,
            colors=(c, c, c),
            linewidth=pred_linewidth,
        )

    # Show ground truth boxes
    for box in boxes_gt_cam:
        c = get_box_color(box.name, is_gt=True)
        box.render(
            ax,
            view=camera_intrinsic,
            normalize=True,
            colors=(c, c, c),
            linewidth=gt_linewidth,
        )

    # Set axis properties
    ax.set_xlim(0, data.size[0])
    ax.set_ylim(data.size[1], 0)
    ax.axis('off')
    # Remove title for individual sensor images
    ax.set_aspect('equal')

    # Save the figure
    if out_path is not None:
        # Create a folder specific to this sensor
        sensor_dir = os.path.join(out_path, sensor_channel)
        os.makedirs(sensor_dir, exist_ok=True)

        # Save in the sensor-specific folder
        sensor_out_path = os.path.join(sensor_dir, f"{sample_token}.png")
        plt.savefig(sensor_out_path, bbox_inches='tight', pad_inches=0.03, dpi=dpi)
    else:
        plt.show()

    plt.close(fig)


def render_sample_data(
    nusc: NuScenes,
    sample_token: str,
    sensor_channels: List[str],
    pred_data: dict,
    out_path: str = None,
    thre: float = 0.15,
    x_center: float = 0.0,
    y_center: float = 0.0,
    gt_linewidth: float = 1.0,
    pred_linewidth: float = 2.0,
) -> None:
    """
    Render sample data with ground truth and predicted boxes for multiple sensors.
    Saves both a combined visualization and individual sensor visualizations.

    :param nusc: NuScenes object
    :param sample_token: Sample token
    :param sensor_channels: List of sensor channels to visualize
    :param pred_data: Prediction data
    :param out_path: Output path for saving the visualization
    :param thre: Detection threshold
    :param x_center: X center for BEV view
    :param y_center: Y center for BEV view
    :param gt_linewidth: Line width for ground truth bounding boxes
    :param pred_linewidth: Line width for prediction bounding boxes
    """
    sample = nusc.get('sample', sample_token)

    # Define camera groups
    vehicle_cams = ['CAM_FRONT', 'CAM_RIGHT', 'CAM_BACK', 'CAM_LEFT']
    air_cams = [
        'CAM_FRONT_AIR',
        'CAM_RIGHT_AIR',
        'CAM_BACK_AIR',
        'CAM_LEFT_AIR',
        'CAM_BOTTOM_AIR',
    ]

    # Check which cameras are available in the sample
    available_vehicle_cams = [cam for cam in vehicle_cams if cam in sample['data']]
    available_air_cams = [cam for cam in air_cams if cam in sample['data']]

    # Get ground truth and prediction boxes
    sample_data_token_lidar = sample['data']['LIDAR_TOP']
    boxes_gt = nusc.get_boxes(sample_data_token_lidar)

    if sample_token in pred_data['results']:
        boxes_pred = [
            Box(
                record['translation'],
                record['size'],
                Quaternion(record['rotation']),
                name=(
                    get_tracking_name(record['tracking_name'])
                    if 'tracking_name' in record
                    else get_tracking_name(record['detection_name'])
                ),
                token=record['tracking_id'] if 'tracking_id' in record else 'predicted',
            )
            for record in pred_data['results'][sample_token]
            if (
                ('tracking_score' in record and record['tracking_score'] > thre)
                or (
                    'tracking_score' not in record
                    and 'detection_score' in record
                    and record['detection_score'] > thre
                )
            )
        ]
    else:
        boxes_pred = []

    # Save individual sensor visualizations
    if out_path is not None:
        # Render and save each sensor individually in separate folders
        all_cams = available_vehicle_cams + available_air_cams
        for sensor_channel in all_cams:
            render_single_sensor(
                nusc,
                sample_token,
                sensor_channel,
                boxes_gt,
                boxes_pred,
                out_path=out_path,
                gt_linewidth=gt_linewidth,
                pred_linewidth=pred_linewidth,
            )

    # Create the combined visualization with 4 rows and 5 columns
    fig = plt.figure(figsize=(24, 12))

    # Create grid layout with 4 rows and 5 columns
    gs = gridspec.GridSpec(4, 5, figure=fig)

    # Create BEV visualization for both GT and Pred
    # Get annotations for BEV visualization
    bbox_gt_list = []
    bbox_pred_list = []

    # Get ground truth boxes for BEV
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        gt_content = nusc.get('sample_annotation', ann)
        tracking_name = get_tracking_name(gt_content['category_name'])
        bbox_gt_list.append(
            TrackingBox(
                sample_token=gt_content['sample_token'],
                translation=tuple(gt_content['translation']),
                size=tuple(gt_content['size']),
                rotation=tuple(gt_content['rotation']),
                velocity=nusc.box_velocity(gt_content['token'])[:2],
                ego_translation=(
                    (0.0, 0.0, 0.0)
                    if 'ego_translation' not in gt_content
                    else tuple(gt_content['ego_translation'])
                ),
                num_pts=(
                    -1 if 'num_pts' not in gt_content else int(gt_content['num_pts'])
                ),
                tracking_name=tracking_name,
                tracking_score=(
                    -1.0
                    if 'tracking_score' not in gt_content
                    else float(gt_content['tracking_score'])
                ),
                tracking_id=gt_content['instance_token'],
            )
        )

    # Get predicted boxes for BEV
    if sample_token in pred_data['results']:
        bbox_anns = pred_data['results'][sample_token]
        for pred_content in bbox_anns:
            tracking_name = get_tracking_name(pred_content['tracking_name'])
            bbox_pred_list.append(
                TrackingBox(
                    sample_token=pred_content['sample_token'],
                    translation=tuple(pred_content['translation']),
                    size=tuple(pred_content['size']),
                    rotation=tuple(pred_content['rotation']),
                    velocity=tuple(pred_content['velocity']),
                    ego_translation=(
                        (0.0, 0.0, 0.0)
                        if 'ego_translation' not in pred_content
                        else tuple(pred_content['ego_translation'])
                    ),
                    num_pts=(
                        -1
                        if 'num_pts' not in pred_content
                        else int(pred_content['num_pts'])
                    ),
                    tracking_name=tracking_name,
                    tracking_score=(
                        -1.0
                        if 'tracking_score' not in pred_content
                        else float(pred_content['tracking_score'])
                    ),
                    tracking_id=pred_content['tracking_id'],
                )
            )

    # Create EvalBoxes objects
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)

    # Eval range for BEV
    eval_range = [-70.0, -40.0, 70.0, 40.0]

    # Render BEV views (GT and Pred)
    ax_bev_gt = fig.add_subplot(gs[0, 4])
    visualize_sample(
        nusc,
        sample_token,
        gt_annotations,
        EvalBoxes(),  # Empty pred boxes for GT view
        conf_th=thre,
        savepath=None,
        eval_range=eval_range,
        ax=ax_bev_gt,
        style='bev',
        x_center=x_center,
        y_center=y_center,
        gt_linewidth=gt_linewidth,
        pred_linewidth=pred_linewidth,
    )
    ax_bev_gt.set_title('BEV (GT)')

    ax_bev_pred = fig.add_subplot(gs[1, 4])
    visualize_sample(
        nusc,
        sample_token,
        EvalBoxes(),  # Empty gt boxes for Pred view
        pred_annotations,
        conf_th=thre,
        savepath=None,
        eval_range=eval_range,
        ax=ax_bev_pred,
        style='bev',
        x_center=x_center,
        y_center=y_center,
        gt_linewidth=gt_linewidth,
        pred_linewidth=pred_linewidth,
    )
    ax_bev_pred.set_title('BEV (Pred)')

    # Function to render a camera view
    def render_camera_view(sensor_channel, row, col):
        # Skip if sensor not available
        if sensor_channel not in sample['data']:
            # Create empty subplot if camera not available
            ax = fig.add_subplot(gs[row, col])
            ax.text(
                0.5,
                0.5,
                f"{sensor_channel} not available",
                horizontalalignment='center',
                verticalalignment='center',
            )
            ax.axis('off')
            return

        # Get the axis for this camera
        ax = plt.subplot(gs[row, col])

        # Get data for this sensor
        data_path, boxes_cam, camera_intrinsic = get_sample_data(
            nusc,
            sample['data'][sensor_channel],
            sensor_channel,
            boxes_gt if row in [0, 2] else boxes_pred,
        )

        if data_path is None:
            ax.text(
                0.5,
                0.5,
                f"{sensor_channel} data not available",
                horizontalalignment='center',
                verticalalignment='center',
            )
            ax.axis('off')
            return

        # Load and display the image
        data = Image.open(data_path)
        ax.imshow(data)

        # Show boxes
        for box in boxes_cam:
            c = get_box_color(box.name, is_gt=(row in [0, 2]))
            box.render(
                ax,
                view=camera_intrinsic,
                normalize=True,
                colors=(c, c, c),
                linewidth=(
                    gt_linewidth if row in [0, 2] else pred_linewidth
                ),  # Use appropriate line width based on whether it's GT or Pred
            )

        # Set axis properties
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)
        ax.axis('off')
        ax.set_title(f"{sensor_channel} {'(GT)' if row in [0, 2] else '(Pred)'}")
        ax.set_aspect('equal')

    # Render vehicle cameras (GT - Row 0, Pred - Row 1)
    for i, cam in enumerate(vehicle_cams):
        render_camera_view(cam, 0, i)  # GT row
        render_camera_view(cam, 1, i)  # Pred row

    # Render air cameras (GT - Row 2, Pred - Row 3)
    for i, cam in enumerate(air_cams):
        render_camera_view(cam, 2, i)  # GT row
        render_camera_view(cam, 3, i)  # Pred row

    plt.tight_layout()

    # Save combined visualization in "combined" folder
    if out_path is not None:
        combined_dir = os.path.join(out_path, 'combined')
        os.makedirs(combined_dir, exist_ok=True)
        combined_out_path = os.path.join(combined_dir, f"{sample_token}.png")
        plt.savefig(combined_out_path, bbox_inches='tight', pad_inches=0.03, dpi=150)
    else:
        plt.show()


def to_video(folder_path, out_path, fps=4, downsample=1, process_sensors=False):
    """
    Create videos from a folder of images.

    :param folder_path: Path to folder containing images
    :param out_path: Path to save the video
    :param fps: Frames per second
    :param downsample: Downsample factor to reduce video size
    :param process_sensors: Whether to also create videos for individual sensors
    """
    # Process combined visualization video (from 'combined' folder)
    combined_dir = os.path.join(folder_path, 'combined')
    if os.path.exists(combined_dir):
        imgs_path = glob.glob(os.path.join(combined_dir, '*.png'))
        imgs_path = sorted(imgs_path)

        if not imgs_path:
            print(f"No images found in {combined_dir}")
        else:
            img_array = []
            for img_path in tqdm(imgs_path, desc="Processing combined frames"):
                img = cv2.imread(img_path)
                height, width, channel = img.shape
                img = cv2.resize(
                    img,
                    (width // downsample, height // downsample),
                    interpolation=cv2.INTER_AREA,
                )
                height, width, channel = img.shape
                size = (width, height)
                img_array.append(img)

            if not img_array:
                print("Failed to load any images for combined view")
            else:
                try:
                    out = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size
                    )
                    for i in range(len(img_array)):
                        out.write(img_array[i])
                    out.release()
                    print(f"Combined video saved to {out_path}")
                except Exception as e:
                    print(f"Error creating combined video: {e}")
    else:
        print(f"Warning: 'combined' directory not found at {combined_dir}")

    # Process individual sensor videos if requested
    if process_sensors:
        # Find all sensor directories
        sensor_dirs = [
            d
            for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d)) and 'CAM_' in d
        ]

        for sensor_name in sensor_dirs:
            sensor_dir = os.path.join(folder_path, sensor_name)
            sensor_imgs = glob.glob(os.path.join(sensor_dir, '*.png'))
            sensor_imgs = sorted(sensor_imgs)

            if not sensor_imgs:
                print(f"No images found for {sensor_name}")
                continue

            img_array = []
            for img_path in tqdm(sensor_imgs, desc=f"Processing {sensor_name} frames"):
                img = cv2.imread(img_path)
                height, width, channel = img.shape
                img = cv2.resize(
                    img,
                    (width // downsample, height // downsample),
                    interpolation=cv2.INTER_AREA,
                )
                height, width, channel = img.shape
                size = (width, height)
                img_array.append(img)

            if not img_array:
                print(f"Failed to load any images for {sensor_name}")
                continue

            try:
                sensor_out_path = out_path.replace('.avi', f'_{sensor_name}.avi')
                out = cv2.VideoWriter(
                    sensor_out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size
                )
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
                print(f"{sensor_name} video saved to {sensor_out_path}")
            except Exception as e:
                print(f"Error creating {sensor_name} video: {e}")


def get_ego_center(nusc, sample_token):
    """
    Calculate ego vehicle center position.

    :param nusc: NuScenes object
    :param sample_token: Sample token
    :return: (x_center, y_center) coordinates
    """
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    ego_position = pose_record['translation']
    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]

    d = 140  # distance from ego to image center
    x_center = ego_position[0] + d * np.cos(yaw)
    y_center = ego_position[1] + d * np.sin(yaw)

    return x_center, y_center


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualization tool for Griffin detection/tracking results'
    )
    parser.add_argument(
        '--predroot',
        required=False,
        help='Path to prediction results_nusc.json. If not provided, only ground truth boxes will be visualized.',
    )
    parser.add_argument('--out_folder', help='Output folder path')
    parser.add_argument('--dataroot', help='Path to dataset root')
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version')
    parser.add_argument(
        '--thre', type=float, default=0.15, help='Detection confidence threshold'
    )
    parser.add_argument(
        '--side',
        choices=['vehicle-side', 'drone-side'],
        default='vehicle-side',
        help='Agent side',
    )
    parser.add_argument(
        '--frequency',
        type=int,
        default=2,
        help='Frequency of frames to visualize (1 = every frame, 2 = every other frame, etc.)',
    )
    parser.add_argument(
        '--skip_video', action='store_true', help='Skip video generation'
    )
    parser.add_argument(
        '--not_skip_sensor_videos',
        action='store_true',
        help='Skip generating individual sensor videos',
    )
    parser.add_argument(
        '--selected_sensors',
        nargs='+',
        default=[
            'CAM_FRONT',
            'CAM_BACK',
            'CAM_LEFT',
            'CAM_RIGHT',
            'CAM_FRONT_AIR',
            'CAM_BACK_AIR',
            'CAM_LEFT_AIR',
            'CAM_RIGHT_AIR',
            'CAM_BOTTOM_AIR',
        ],
        help='Selected sensors to visualize',
    )
    parser.add_argument(
        '--target_samples',
        nargs='+',
        default=None,
        help='List of specific sample tokens to visualize (e.g., sample_1735724827300000)',
    )
    parser.add_argument(
        '--gt_linewidth',
        type=float,
        default=1.5,
        help='Line width for ground truth bounding boxes',
    )
    parser.add_argument(
        '--pred_linewidth',
        type=float,
        default=1.5,
        help='Line width for prediction bounding boxes',
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print(f"Processing Griffin dataset {args.side}, writing to {args.out_folder}")

    # Create output folder
    os.makedirs(args.out_folder, exist_ok=True)

    # Create "combined" folder for combined visualizations
    combined_dir = os.path.join(args.out_folder, 'combined')
    os.makedirs(combined_dir, exist_ok=True)

    # Define video path
    video_path = os.path.join(
        os.path.dirname(args.out_folder),
        f'{args.side}-{os.path.basename(args.out_folder)}.avi',
    )

    # Initialize prediction results
    if args.predroot:
        # Load predictions
        try:
            pred_results = mmcv.load(args.predroot)
            print(f"Loaded predictions from {args.predroot}")
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return
    else:
        # Create empty prediction dictionary for ground truth only visualization
        pred_results = {'results': {}}
        print("No prediction data provided. Visualizing ground truth only.")

    # Initialize nuScenes
    try:
        nusc = NuScenes(
            version=args.version,
            dataroot=args.dataroot,
            verbose=True,
        )
        print(f"Initialized nuScenes with {len(nusc.sample)} samples")
    except Exception as e:
        print(f"Error initializing nuScenes: {e}")
        return

    # Get sample tokens
    if args.target_samples:
        # Use the provided target sample tokens
        sample_token_list = args.target_samples
        print(f"Using {len(sample_token_list)} target samples for visualization")

        # Make sure all target samples are in prediction results (if using predictions)
        if args.predroot and 'results' in pred_results:
            for token in sample_token_list:
                if token not in pred_results['results']:
                    print(f"Warning: Sample {token} not found in prediction results")
                    pred_results['results'][token] = []
        else:
            # Add empty prediction entries for each sample
            for token in sample_token_list:
                pred_results['results'][token] = []
    elif args.predroot and 'results' in pred_results:
        # Get sample tokens from prediction results
        sample_token_list = list(pred_results['results'].keys())[:: args.frequency]
        # sample_token_list = sample_token_list[:2]  # only for testing
        print(f"Found {len(sample_token_list)} samples with predictions to visualize")
    else:
        # Get sample tokens directly from nuScenes
        sample_token_list = [s['token'] for s in nusc.sample][:: args.frequency]
        # Add empty prediction entries for each sample
        for token in sample_token_list:
            pred_results['results'][token] = []
        print(
            f"Using {len(sample_token_list)} samples from dataset for ground truth visualization"
        )

    if not sample_token_list:
        print("No samples to visualize")
        return

    # Get ego vehicle center for first frame
    try:
        x_center, y_center = get_ego_center(nusc, sample_token_list[0])
    except Exception as e:
        print(f"Error getting ego center: {e}, using defaults")
        x_center, y_center = 0.0, 0.0

    # Render each frame
    for i, sample_token in enumerate(tqdm(sample_token_list, desc="Rendering frames")):
        try:
            render_sample_data(
                nusc=nusc,
                sample_token=sample_token,
                sensor_channels=args.selected_sensors,
                pred_data=pred_results,
                out_path=args.out_folder,
                thre=args.thre,
                x_center=x_center,
                y_center=y_center,
                gt_linewidth=args.gt_linewidth,
                pred_linewidth=args.pred_linewidth,
            )
        except Exception as e:
            print(f"Error rendering sample {sample_token}: {e}")

    # Generate video
    if not args.skip_video:
        try:
            to_video(
                folder_path=args.out_folder,
                out_path=video_path,
                process_sensors=args.not_skip_sensor_videos,
            )
        except Exception as e:
            print(f"Error generating video: {e}")


if __name__ == '__main__':
    main()

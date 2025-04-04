import mmcv
import argparse
import os
import glob
import cv2
import copy
from nuscenes.nuscenes import NuScenes
from PIL import Image
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import (
    LidarPointCloud,
    Box,
    PointCloud,
)
from nuscenes.utils.geometry_utils import (
    view_points,
    box_in_image,
    BoxVisibility,
    transform_matrix,
)
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.utils import boxes_to_sensor

# Initialize TRACKING_NAMES with Griffin dataset class names to fix the assertion error
# This is necessary because the TRACKING_NAMES list is empty by default
from nuscenes.eval.tracking.data_classes import TRACKING_NAMES

# Define valid tracking names for Griffin dataset
VALID_TRACKING_NAMES = [
    'car',
    # 'truck',
    # 'trailer',
    # 'bus',
    # 'construction_vehicle',
    'bicycle',
    # 'motorcycle',
    'pedestrian',
    # 'traffic_cone',
    # 'barrier',
]

# Set TRACKING_NAMES globally to allow TrackingBox to validate correctly
TRACKING_NAMES[:] = VALID_TRACKING_NAMES

# Colorful bounding box colors for different tracking IDs
id_colors = [
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
    Get the appropriate color for a bounding box based on its class and source.

    Selects a color from the predefined color scheme based on whether the box
    is from ground truth or predictions, and the object class.

    Args:
        box_name: Name/class of the box (e.g., 'car', 'bicycle', 'pedestrian')
        is_gt: Whether the box is ground truth (True) or prediction (False)

    Returns:
        RGB or RGBA color array to use for the box
    """
    # Determine the source (gt or pred)
    source = 'gt' if is_gt else 'pred'

    # Convert box name to lowercase for consistent matching
    name_lower = box_name.lower() if box_name else ''

    # Match to known classes
    if 'car' in name_lower:
        return class_colors['car'][source]
    elif 'bicycle' in name_lower:
        return class_colors['bicycle'][source]
    elif 'pedestrian' in name_lower:
        return class_colors['pedestrian'][source]
    else:
        return class_colors['default'][source]


def render_bev_view(
    nusc: NuScenes,
    sample_token: str,
    eval_boxes: EvalBoxes,
    is_gt: bool,
    conf_th: float = 0.15,
    eval_range: list = [-70.0, -40.0, 70.0, 40.0],
    ax=None,
    style: str = 'bev',
    linewidth: float = 1.5,
) -> None:
    """
    Render a Bird's Eye View (BEV) visualization with boxes.

    Creates a top-down view showing objects represented as 2D bounding boxes,
    with optional filtering based on confidence thresholds for predictions.

    Args:
        nusc: NuScenes object with dataset information
        sample_token: Sample token to visualize
        eval_boxes: Boxes grouped by sample (from EvalBoxes class)
        is_gt: Whether the boxes are ground truth (True) or predictions (False)
        conf_th: Confidence threshold to filter predictions (ignored for GT boxes)
        eval_range: Visualization range in meters [xmin, ymin, xmax, ymax]
        ax: Matplotlib axes on which to render (created if None)
        style: Visualization style - 'bev' for sensor frame, 'utm' for global frame
        linewidth: Line width for bounding boxes
    """
    # Retrieve sensor & pose records
    sample_rec = nusc.get('sample', sample_token)
    lidar_sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    lidar_cs_record = nusc.get(
        'calibrated_sensor', lidar_sd_record['calibrated_sensor_token']
    )
    ego_pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])

    # Get boxes for this sample
    boxes_global = eval_boxes[sample_token]

    # Convert global boxes to ego top lidar coordinate if in BEV mode
    if style == 'bev':
        boxes_lidar = boxes_to_sensor(boxes_global, ego_pose_record, lidar_cs_record)

    # Initialize visualization
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show ego vehicle position
    if style == 'bev':
        ax.plot(0, 0, 'x', color='black')  # Origin in sensor frame
    elif style == 'utm':
        ego_position = ego_pose_record['translation']
        ax.plot(ego_position[0], ego_position[1], 'x', color='black')  # Global position

    # Render boxes with appropriate colors
    for box, box_global in zip(boxes_lidar, boxes_global):
        # Filter predictions by confidence threshold
        if not is_gt:
            assert not np.isnan(
                box_global.tracking_score
            ), 'Error: Box score cannot be NaN!'
            if box_global.tracking_score < conf_th:
                continue

        # Select box color based on class or tracking ID
        if hasattr(box_global, 'tracking_name'):
            # Use class-based coloring
            c = get_box_color(box_global.tracking_name, is_gt=is_gt)
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=linewidth)
        else:
            # Fallback to tracking ID based coloring
            tr_id = box_global.tracking_id
            c = id_colors[tr_id % len(id_colors)]
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=linewidth)

    # Limit visible range
    ax.set_xlim(eval_range[0], eval_range[2])
    ax.set_ylim(eval_range[1], eval_range[3])

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


def render_sample_data(
    nusc: NuScenes,
    sample_token: str,
    pred_data: dict,
    out_path: str = None,
    thre: float = 0.15,
    linewidth: float = 1.5,
) -> None:
    """
    Render a comprehensive visualization of a sample with ground truth and predicted boxes.

    Creates a combined visualization with multiple camera views and BEV perspectives,
    arranged in a grid layout with the following structure:
    - Row 0: Ground truth boxes on vehicle cameras (Front, Right, Back, Left)
    - Row 1: Prediction boxes on vehicle cameras
    - Row 2: Ground truth boxes on drone cameras (Front, Right, Back, Left, Bottom)
    - Row 3: Prediction boxes on drone cameras
    - Column 4 (up-right side): BEV views for ground truth and predictions

    Args:
        nusc: NuScenes object with dataset information
        sample_token: Sample token to visualize
        pred_data: Dictionary containing prediction results
        out_path: Output path for saving visualization images
        thre: Detection threshold for filtering low-confidence predictions
        linewidth: Line width for bounding box visualization
    """
    sample = nusc.get('sample', sample_token)

    # Define camera groups for vehicle and drone perspectives
    vehicle_cams = ['CAM_FRONT', 'CAM_RIGHT', 'CAM_BACK', 'CAM_LEFT']
    air_cams = [
        'CAM_FRONT_AIR',
        'CAM_RIGHT_AIR',
        'CAM_BACK_AIR',
        'CAM_LEFT_AIR',
        'CAM_BOTTOM_AIR',
    ]

    # Get ground truth boxes
    bbox_gt_list = []
    for ann in sample['anns']:
        gt_content = nusc.get('sample_annotation', ann)
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
                    -1
                    if 'num_lidar_pts' not in gt_content
                    else int(gt_content['num_lidar_pts'])
                ),
                tracking_name=gt_content['category_name'],
                tracking_score=1.0,
                tracking_id=int(gt_content['instance_token'].split('_')[-1]),
            )
        )

    # Get predicted boxes
    bbox_pred_list = []
    if sample_token in pred_data['results']:
        bbox_anns = pred_data['results'][sample_token]
        for pred_content in bbox_anns:
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
                    tracking_name=pred_content['tracking_name'],
                    tracking_score=(
                        -1.0
                        if 'tracking_score' not in pred_content
                        else float(pred_content['tracking_score'])
                    ),
                    tracking_id=pred_content['tracking_id'],
                )
            )

    # Create EvalBoxes objects for visualization
    gt_eval_boxes = EvalBoxes()
    pred_eval_boxes = EvalBoxes()
    gt_eval_boxes.add_boxes(sample_token, bbox_gt_list)
    pred_eval_boxes.add_boxes(sample_token, bbox_pred_list)

    # Set up the visualization grid
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(4, 5, figure=fig)

    # Define BEV visualization range
    eval_range = [-70.0, -40.0, 70.0, 40.0]

    # Render BEV views (GT)
    ax_bev_gt = fig.add_subplot(gs[0, 4])
    render_bev_view(
        nusc,
        sample_token,
        gt_eval_boxes,
        is_gt=True,
        conf_th=thre,
        eval_range=eval_range,
        ax=ax_bev_gt,
        style='bev',
        linewidth=linewidth,
    )
    ax_bev_gt.set_title('BEV (GT)')

    # Render BEV views (Pred)
    ax_bev_pred = fig.add_subplot(gs[1, 4])
    render_bev_view(
        nusc,
        sample_token,
        pred_eval_boxes,
        is_gt=False,
        conf_th=thre,
        eval_range=eval_range,
        ax=ax_bev_pred,
        style='bev',
        linewidth=linewidth,
    )
    ax_bev_pred.set_title('BEV (Pred)')

    def render_camera_view(sensor_channel, boxes_global, row, col):
        """Helper function to render a single camera view in the grid"""
        # Get the axis for this camera
        ax = plt.subplot(gs[row, col])

        # Skip if sensor not available
        if sensor_channel not in sample['data']:
            # Create empty subplot if camera not available
            ax.text(
                0.5,
                0.5,
                f"{sensor_channel} not available",
                horizontalalignment='center',
                verticalalignment='center',
            )
            ax.axis('off')
            return

        # Retrieve sensor & pose records
        sensor_sd_record = nusc.get('sample_data', sample['data'][sensor_channel])
        sensor_cs_record = nusc.get(
            'calibrated_sensor', sensor_sd_record['calibrated_sensor_token']
        )
        ego_pose_record = nusc.get('ego_pose', sensor_sd_record['ego_pose_token'])

        # Get data for this sensor
        data_path = nusc.get_sample_data_path(sensor_sd_record['token'])
        boxes_cam = boxes_to_sensor(boxes_global, ego_pose_record, sensor_cs_record)
        camera_intrinsic = np.array(sensor_cs_record['camera_intrinsic'])

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
        for box, box_global in zip(boxes_cam, boxes_global):
            if not box_in_image(
                box,
                camera_intrinsic,
                (data.size[0], data.size[1]),
                vis_level=BoxVisibility.ANY,
            ):
                continue
            c = get_box_color(box_global.tracking_name, is_gt=(row in [0, 2]))
            box.render(
                ax,
                view=camera_intrinsic,
                normalize=True,
                colors=(c, c, c),
                linewidth=linewidth,
            )

        # Set axis properties
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)
        ax.axis('off')
        ax.set_title(f"{sensor_channel} {'(GT)' if row in [0, 2] else '(Pred)'}")
        ax.set_aspect('equal')

    # Render vehicle cameras (GT - Row 0, Pred - Row 1)
    for i, cam in enumerate(vehicle_cams):
        render_camera_view(cam, gt_eval_boxes[sample_token], 0, i)  # GT row
        render_camera_view(cam, pred_eval_boxes[sample_token], 1, i)  # Pred row

    # Render air cameras (GT - Row 2, Pred - Row 3)
    for i, cam in enumerate(air_cams):
        render_camera_view(cam, gt_eval_boxes[sample_token], 2, i)  # GT row
        render_camera_view(cam, pred_eval_boxes[sample_token], 3, i)  # Pred row

    plt.tight_layout()

    # Save combined visualization in "combined" folder
    if out_path is not None:
        combined_out_path = os.path.join(out_path, "combined", f"{sample_token}.png")
        plt.savefig(combined_out_path, bbox_inches='tight', pad_inches=0.03, dpi=150)
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()


def to_video(folder_path, out_path, fps=4, downsample=1):
    """
    Create a video from a sequence of visualization images.

    Searches for rendered images in the 'combined' subfolder of the given folder path,
    then compiles them into a video file using OpenCV.

    Args:
        folder_path: Path to the folder containing image subfolders
        out_path: Path to save the output video file
        fps: Frames per second for the output video (default: 4)
        downsample: Downsample factor to reduce video resolution (default: 1, no downsampling)
    """
    # Process combined visualization video (from 'combined' folder)
    combined_dir = os.path.join(folder_path, 'combined')
    if os.path.exists(combined_dir):
        # Find and sort all PNG images in the directory
        imgs_path = glob.glob(os.path.join(combined_dir, '*.png'))
        imgs_path = sorted(imgs_path)

        if not imgs_path:
            print(f"No images found in {combined_dir}")
        else:
            # Process each image frame
            img_array = []
            for img_path in tqdm(imgs_path, desc="Processing combined frames"):
                # Read and resize image
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
                    # Create video writer and write frames
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


def parse_args():
    """
    Parse command line arguments for the Griffin visualization tool.

    Returns:
        args: Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description='Visualization tool for Griffin detection/tracking results'
    )
    # Required and important arguments
    parser.add_argument(
        '--dataroot',
        required=True,
        help='Path to dataset root (choose early fusion set to get both vehicle and drone side images)',
    )
    parser.add_argument(
        '--out_folder',
        required=True,
        help='Output folder path for visualization results',
    )
    parser.add_argument(
        '--predroot',
        required=False,
        help='Path to prediction results JSON file. If not provided, only ground truth boxes will be visualized.',
    )

    # Dataset configuration
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version')

    # Visualization settings
    parser.add_argument(
        '--thre', type=float, default=0.15, help='Detection confidence threshold'
    )
    parser.add_argument(
        '--frequency',
        type=int,
        default=2,
        help='Frequency of frames to visualize (1 = every frame, 2 = every other frame, etc.)',
    )
    parser.add_argument(
        '--linewidth', type=float, default=1.5, help='Line width for bounding boxes'
    )

    # Output options
    parser.add_argument(
        '--skip_video', action='store_true', help='Skip video generation'
    )

    # Sample selection
    parser.add_argument(
        '--target_samples',
        nargs='+',
        default=None,
        help='List of specific sample tokens to visualize (e.g., sample_1735724827300000)',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(f"Processing Griffin dataset, writing to {args.out_folder}")

    # Setup output directories
    combined_dir = os.path.join(args.out_folder, 'combined')
    os.makedirs(combined_dir, exist_ok=True)

    # Define video path
    video_path = os.path.join(
        args.out_folder,
        f'{os.path.basename(args.out_folder)}.avi',
    )

    # Load prediction results if provided
    if args.predroot:
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

    # Initialize dataset
    try:
        nusc = NuScenes(
            version=args.version,
            dataroot=args.dataroot,
            verbose=True,
        )
        print(f"Initialized dataset with {len(nusc.sample)} samples")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    # Determine samples to visualize
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
        # sample_token_list = sample_token_list[:3]  # only for debug
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

    # Generate visualizations for each sample
    for i, sample_token in enumerate(tqdm(sample_token_list, desc="Rendering frames")):
        try:
            render_sample_data(
                nusc=nusc,
                sample_token=sample_token,
                pred_data=pred_results,
                out_path=args.out_folder,
                thre=args.thre,
                linewidth=args.linewidth,
            )
        except Exception as e:
            print(f"Error rendering sample {sample_token}: {e}")

    # Generate video (if requested)
    if not args.skip_video:
        try:
            to_video(
                folder_path=args.out_folder,
                out_path=video_path,
            )
        except Exception as e:
            print(f"Error generating video: {e}")


if __name__ == '__main__':
    main()

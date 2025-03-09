import os
import math
import pickle
import numpy as np
from tqdm import tqdm

from utils import (
    read_json,
    trans_point,
    get_lidar2camera,
    get_cam_calib_intrinsic,
    get_label_lidar_rotation,
    get_camera_3d_alpha_rotation,
)
import argparse

type2id = {
    "Car": 2,
    "Van": 2,
    "Truck": 2,
    "Bus": 2,
    "Cyclist": 1,
    "Tricyclist": 3,
    "Motorcyclist": 3,
    "Barrow": 3,
    "Barrowlist": 3,
    "Pedestrian": 0,
    "Trafficcone": 3,
    "Pedestrianignore": 3,
    "Carignore": 3,
    "otherignore": 3,
    "unknowns_unmovable": 3,
    "unknowns_movable": 3,
    "unknown_unmovable": 3,
    "unknown_movable": 3,
}

# id2type = {0: "Pedestrian", 1: "Cyclist", 2: "Car", 3: "Motorcyclist"}
id2type = {0: "Car", 1: "Bicycle", 2: "Pedestrian"}


def get_sequence_id(frame, data_info):
    for obj in data_info:
        if frame == obj["frame_id"]:
            sequence_id = obj["sequence_id"]
            return sequence_id


def get_sequence_id_griffin(frame, sample_info, scene_info):
    for sample in sample_info:
        if frame == sample["token"]:
            scene_token = sample["scene_token"]
            for scene in scene_info:
                if scene_token == scene["token"]:
                    return scene["name"]
    raise ValueError(f"No scene token found for frame: {frame}")


def trans_points_cam2img(camera_3d_8points, calib_intrinsic, with_depth=False):
    """
    Transform points from camera coordinates to image coordinates.
    Args:
        camera_3d_8points: list(8, 3)
        calib_intrinsic: np.array(3, 4)
    Returns:
        list(8, 2)
    """
    camera_3d_8points = np.array(camera_3d_8points)
    points_shape = np.array([8, 1])
    points_4 = np.concatenate((camera_3d_8points, np.ones(points_shape)), axis=-1)
    point_2d = np.dot(calib_intrinsic, points_4.T)
    point_2d = point_2d.T
    point_2d_res = point_2d[:, :2] / point_2d[:, 2:3]
    if with_depth:
        return np.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res.tolist()


def get_calib_griffin(calib_path):
    from nuscenes.eval.common.utils import Quaternion

    calibs = read_json(calib_path)
    for calib in calibs:
        if calib['token'] == 'calibrated_sensor_front':
            rotation = Quaternion(calib['lidar2cam_rotation']).rotation_matrix
            translation = np.array(calib['lidar2cam_translation']).reshape(3, 1)
            camera_intrinsic = np.zeros([3, 4])
            camera_intrinsic[:3, :3] = np.array(calib['camera_intrinsic']).reshape(3, 3)
            return rotation, translation, camera_intrinsic
    raise ValueError(f"No calibrated_sensor_front in {calib_path}")


def label_det_result2kitti(input_file_path, output_dir_path, data_root):
    """
    Convert detection results from mmdetection3d_kitti format to KITTI format.
    Args:
        input_file_path: mmdetection3d_kitti results pickle file path
        output_dir_path: converted kitti format file directory
        data_root: path to dataset
    """
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    with open(input_file_path, 'rb') as load_f:
        bbox_results = pickle.load(load_f)['bbox_results']

    if "spd" in data_root or "SPD" in data_root:
        # SPD Warning: check whether the sensor calibration is the same along the sequence, temporarily use the first sample token
        sample_token = bbox_results[0]['token']
        veh_frame = sample_token
        lidar2camera_path = (
            f'{data_root}/vehicle-side/calib/lidar_to_camera/{veh_frame}.json'
        )
        camera2image_path = (
            f'{data_root}/vehicle-side/calib/camera_intrinsic/{veh_frame}.json'
        )
        rotation, translation = get_lidar2camera(lidar2camera_path)
        calib_intrinsic = get_cam_calib_intrinsic(camera2image_path)
    elif "griffin" in data_root:
        calib_path = f'{data_root}/vehicle-side/v1.0-trainval/calibrated_sensor.json'
        rotation, translation, calib_intrinsic = get_calib_griffin(calib_path)
    else:
        raise NotImplementedError(f"Invalid data root path: {data_root}")

    for bbox_result in tqdm(bbox_results, desc="Converting detection results"):
        sample_token = bbox_result['token']
        veh_frame = sample_token
        if "griffin" in data_root:
            veh_frame = float(veh_frame.split('_')[-1]) / 1e5

        output_file_path = output_dir_path + '/' + sample_token + '.txt'
        if os.path.exists(output_file_path):
            print(
                "veh_frame",
                veh_frame,
                "det_result_name",
                input_file_path.split('/')[-1].split('.')[0],
            )
            save_file = open(output_file_path, 'a')
        else:
            save_file = open(output_file_path, 'w')
        num_boxes = len(bbox_result["labels_3d"].tolist())
        for i in range(num_boxes):
            box_3d = bbox_result["boxes_3d"][i]
            lidar_3d_8points_det_result = box_3d.corners.tolist()[0]
            lidar_3d_8points = [
                lidar_3d_8points_det_result[3],
                lidar_3d_8points_det_result[0],
                lidar_3d_8points_det_result[4],
                lidar_3d_8points_det_result[7],
                lidar_3d_8points_det_result[2],
                lidar_3d_8points_det_result[1],
                lidar_3d_8points_det_result[5],
                lidar_3d_8points_det_result[6],
            ]

            # calculate l, w, h, x, y, z in LiDAR coordinate system
            lidar_xy0, lidar_xy3, lidar_xy1 = (
                lidar_3d_8points[0][0:2],
                lidar_3d_8points[3][0:2],
                lidar_3d_8points[1][0:2],
            )
            lidar_z4, lidar_z0 = lidar_3d_8points[4][2], lidar_3d_8points[0][2]
            l = math.sqrt(
                (lidar_xy0[0] - lidar_xy3[0]) ** 2 + (lidar_xy0[1] - lidar_xy3[1]) ** 2
            )
            w = math.sqrt(
                (lidar_xy0[0] - lidar_xy1[0]) ** 2 + (lidar_xy0[1] - lidar_xy1[1]) ** 2
            )
            h = lidar_z4 - lidar_z0
            lidar_x0, lidar_y0 = lidar_3d_8points[0][0], lidar_3d_8points[0][1]
            lidar_x2, lidar_y2 = lidar_3d_8points[2][0], lidar_3d_8points[2][1]
            lidar_x = (lidar_x0 + lidar_x2) / 2
            lidar_y = (lidar_y0 + lidar_y2) / 2
            lidar_z = (lidar_z0 + lidar_z4) / 2

            # assert box_3d.dims.tolist() == [l, w, h]
            assert np.allclose(box_3d.dims.tolist(), [l, w, h], atol=1e-3)
            assert np.allclose(
                box_3d.center[0], [lidar_x, lidar_y, lidar_z - h / 2], atol=1e-3
            )

            obj_type = id2type[bbox_result["labels_3d"].tolist()[i]]
            score = bbox_result["scores_3d"].tolist()[i]

            camera_3d_8points = []
            for lidar_point in lidar_3d_8points:
                camera_point = trans_point(lidar_point, rotation, translation)
                camera_3d_8points.append(camera_point)

            # # generate the yaw angle of the object in the lidar coordinate system at the vehicle-side.
            # lidar_rotation = get_label_lidar_rotation(lidar_3d_8points)
            lidar_rotation = box_3d.yaw.tolist()[0]
            # generate the alpha and yaw angle of the object in the camera coordinate system at the vehicle-side
            camera_x0, camera_z0 = camera_3d_8points[0][0], camera_3d_8points[0][2]
            camera_x2, camera_z2 = camera_3d_8points[2][0], camera_3d_8points[2][2]
            camera_x = (camera_x0 + camera_x2) / 2
            camera_y = camera_3d_8points[0][1]
            camera_z = (camera_z0 + camera_z2) / 2
            camera_3d_location = [camera_x, camera_y, camera_z]

            image_8points_2d = trans_points_cam2img(camera_3d_8points, calib_intrinsic)
            x_max = max(image_8points_2d[:][0])
            x_min = min(image_8points_2d[:][0])
            y_max = max(image_8points_2d[:][1])
            y_min = min(image_8points_2d[:][1])

            alpha, camera_rotation = get_camera_3d_alpha_rotation(
                camera_3d_8points, camera_3d_location
            )

            str_item = (
                str(veh_frame)
                + ' '
                + str(obj_type)
                + ' '
                + '-1'
                + ' '
                + '-1'
                + ' '
                + '-1'
                + ' '
                + str(alpha)
                + ' '
                + str(x_min)
                + ' '
                + str(y_min)
                + ' '
                + str(x_max)
                + ' '
                + str(y_max)
                + ' '
                + str(h)
                + ' '
                + str(w)
                + ' '
                + str(l)
                + ' '
                + str(camera_x)
                + ' '
                + str(camera_y)
                + ' '
                + str(camera_z)
                + ' '
                + str(camera_rotation)
                + ' '
                + str(lidar_x)
                + ' '
                + str(lidar_y)
                + ' '
                + str(lidar_z)
                + ' '
                + str(lidar_rotation)
                + ' '
                + '-1'
                + ' '
                + str(score)
                + ' '
                + '-1'
                + ' '
                + '-1'
                + '\n'
            )
            save_file.writelines(str_item)
        save_file.close()


def gen_kitti_result(input_pkl_path, output_dir_path, data_root):
    """
    Convert detection results from mmdetection3d_kitti format to KITTI format for input pkl file.
    Args:
        input_pkl_path: path to mmdetection3d_kitti results pickle file
        output_dir_path: directory to save converted KITTI format files
        data_root: path to dataset
    """
    if os.path.exists(output_dir_path):
        os.system('rm -rf %s' % output_dir_path)
    os.makedirs(output_dir_path, exist_ok=True)
    label_det_result2kitti(input_pkl_path, output_dir_path, data_root)


def gen_kitti_seq_result(input_dir_path, output_dir_path, data_root):
    """
    Convert detection results from mmdetection3d_kitti format to KITTI format and group them by sequence.
    Args:
        input_dir_path: directory containing converted KITTI format files
        output_dir_path: directory to save converted KITTI format files grouped by sequence
        data_root: path to dataset
    """
    if "spd" in data_root or "SPD" in data_root:
        data_info = read_json(f'{data_root}/vehicle-side/data_info.json')
    elif "griffin" in data_root:
        sample_info = read_json(f'{data_root}/vehicle-side/v1.0-trainval/sample.json')
        scene_info = read_json(f'{data_root}/vehicle-side/v1.0-trainval/scene.json')
    else:
        raise NotImplementedError(f"Invalid data root path: {data_root}")

    if os.path.exists(output_dir_path):
        os.system('rm -rf %s' % output_dir_path)
    os.makedirs(output_dir_path)

    list_input_files = os.listdir(input_dir_path)
    for input_file in tqdm(
        list_input_files, desc="Grouping detection results by sequence"
    ):
        input_file_path = input_dir_path + '/' + input_file
        if "spd" in data_root or "SPD" in data_root:
            sequence_id = get_sequence_id(input_file.split('.')[0], data_info)
        elif "griffin" in data_root:
            sequence_id = get_sequence_id_griffin(
                input_file.split('.')[0], sample_info, scene_info
            )

        sequence_path = output_dir_path + '/' + sequence_id
        os.makedirs(sequence_path, exist_ok=True)
        os.system('cp %s %s/' % (input_file_path, sequence_path))


def gen_kitti_seq_txt(input_dir_path, output_dir_path):
    """
    Group converted KITTI format files by sequence and write them into one txt file per sequence.
    Args:
        input_dir_path: directory containing KITTI format files grouped by sequence
        output_dir_path: directory to save txt files grouped by sequence
    """
    if os.path.exists(output_dir_path):
        os.system('rm -rf %s' % output_dir_path)
    os.makedirs(output_dir_path)
    list_dir_sequences = os.listdir(input_dir_path)
    for dir_sequence in tqdm(
        list_dir_sequences, desc="Merging sequence detection results"
    ):
        path_seq_input = input_dir_path + '/' + dir_sequence
        file_output = output_dir_path + '/' + dir_sequence + '.txt'
        save_file = open(file_output, 'w')
        list_files = os.listdir(path_seq_input)
        list_files.sort()
        for file in list_files:
            path_file = path_seq_input + '/' + file
            with open(path_file, "r") as read_f:
                data_txt = read_f.readlines()
                for item in data_txt:
                    save_file.writelines(item)
        save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert detection results to KITTI format'
    )
    parser.add_argument(
        '--input-pkl-path',
        type=str,
        default='projects/work_dirs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion/results.pkl',
        help='Path to mmdetection3d_kitti results pickle file',
    )
    parser.add_argument(
        '--output-dir-path',
        type=str,
        default='projects/work_dirs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion/detection_results_to_kitti',
        help='Directory to save converted KITTI format files',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='datasets/griffin_50scenes_25m/griffin-nuscenes',
        help='Path to SPD dataset',
    )
    args = parser.parse_args()

    input_pkl_path = args.input_pkl_path
    data_root = args.data_root
    print(f"Transferring detection results to KITTI format...")
    print(f"Input pkl path: {input_pkl_path}")
    print(f"Output dir path: {args.output_dir_path}")
    print(f"Data root: {data_root}")

    output_dir_path = os.path.join(args.output_dir_path, 'label')
    output_dir_path_seq = os.path.join(args.output_dir_path, 'label_seq')
    output_dir_path_track = os.path.join(args.output_dir_path + 'label_track')
    # Convert detection results from mmdetection3d_kitti format to KITTI format for all files in input_dir_path
    gen_kitti_result(input_pkl_path, output_dir_path, data_root)
    # Group converted KITTI format files by sequence
    gen_kitti_seq_result(output_dir_path, output_dir_path_seq, data_root)
    # Group converted KITTI format files by sequence and write them into one txt file per sequence
    gen_kitti_seq_txt(output_dir_path_seq, output_dir_path_track)

    print(f"Copying detection results to output dir...")
    os.system("cp %s/* %s/" % (output_dir_path_track, args.output_dir_path))
    os.system("rm -rf %s" % (output_dir_path))
    os.system("rm -rf %s" % (output_dir_path_seq))
    os.system("rm -rf %s" % (output_dir_path_track))

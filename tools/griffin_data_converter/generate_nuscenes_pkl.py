import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import Quaternion
import mmcv
import numpy as np
import os
import json


def create_nuscenes_infos(
    root_path, out_path, info_prefix, version, side, split_info, delay_frame_num=0
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
        side (str, optional): Side out of ["drone", "vehicle", "cooperative", "early-fusion"].
    """
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    train_scenes = split_info['train']
    val_scenes = split_info['val']

    scene_names = [s["name"] for s in nusc.scene]
    train_scenes = set(
        [nusc.scene[scene_names.index(s)]["token"] for s in train_scenes]
    )
    val_scenes = set([nusc.scene[scene_names.index(s)]["token"] for s in val_scenes])

    # TODO: No Test Scenes
    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, side, delay_frame_num
    )

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = "{}_infos_test.pkl".format(info_prefix)
        mmcv.dump(data, info_path)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = os.path.join(out_path, "{}_infos_train.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = os.path.join(out_path, "{}_infos_val.pkl".format(info_prefix))
        mmcv.dump(data, info_val_path)


def _fill_trainval_infos(
    nusc: NuScenes, train_scenes, val_scenes, test=False, side=None, delay_frame_num=0
):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        side (str, optional): Side out of ["drone", "vehicle", "cooperative"].
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        lidar_calib = nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        pose_record = nusc.get("ego_pose", lidar_data["ego_pose_token"])

        info = {
            "token": sample["token"],
            # "sweeps": [],
            "cams": {},
            "lidar2ego_translation": lidar_calib["translation"],
            "lidar2ego_rotation": lidar_calib["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            # For v2x-seq begin
            "frame_idx": sample["frame_idx"],
            "scene_token": sample['scene_token'],
            "prev": sample['prev'],
            "next": sample['next'],
            # For v2x-seq end
        }

        if side in ["cooperative", "early-fusion"]:
            delay_sample = sample
            if delay_frame_num > 0:
                try:
                    for i in range(delay_frame_num):
                        assert (
                            delay_sample["prev"] != ""
                        ), f"Not enough previous sample for {sample['token']}, skip this sample"
                        delay_sample = nusc.get("sample", delay_sample["prev"])
                except AssertionError as e:
                    # print(e)
                    continue
            delay_sample_token = delay_sample["token"]
            delay_pose_record = nusc.get("ego_pose", delay_sample["ego_pose_token"])

            info["air_sample_token"] = delay_sample_token

        if side == "cooperative":
            airLidar_token = delay_sample["data"]["LIDAR_TOP_AIR"]
            airLidar_data = nusc.get("sample_data", airLidar_token)
            airLidar_calib = nusc.get(
                "calibrated_sensor", airLidar_data["calibrated_sensor_token"]
            )

            T_airLidar_to_air = np.eye(4)
            T_airLidar_to_air[:3, :3] = Quaternion(
                airLidar_calib["rotation"]
            ).rotation_matrix
            T_airLidar_to_air[:3, 3] = airLidar_calib["translation"]
            T_air_to_airLidar = np.linalg.inv(T_airLidar_to_air)

            T_air_to_global = np.eye(4)
            T_air_to_global[:3, :3] = Quaternion(
                delay_pose_record["air_rotation"]
            ).rotation_matrix
            T_air_to_global[:3, 3] = delay_pose_record["air_translation"]
            T_global_to_air = np.linalg.inv(T_air_to_global)

            T_ego_to_global = np.eye(4)
            T_ego_to_global[:3, :3] = Quaternion(
                pose_record["rotation"]
            ).rotation_matrix
            T_ego_to_global[:3, 3] = pose_record["translation"]

            T_lidar_to_ego = np.eye(4)
            T_lidar_to_ego[:3, :3] = Quaternion(lidar_calib["rotation"]).rotation_matrix
            T_lidar_to_ego[:3, 3] = lidar_calib["translation"]

            T_vehLidar_to_airLidar = (
                T_air_to_airLidar @ T_global_to_air @ T_ego_to_global @ T_lidar_to_ego
            )
            info['vehLidar2airLidar_rt'] = T_vehLidar_to_airLidar.tolist()

        # obtain images' information per frame
        if side == "drone":
            camera_types = [
                "CAM_FRONT",
                "CAM_BACK",
                "CAM_LEFT",
                "CAM_RIGHT",
                "CAM_BOTTOM",
            ]
        elif side == "vehicle":
            camera_types = [
                "CAM_FRONT",
                "CAM_BACK",
                "CAM_LEFT",
                "CAM_RIGHT",
            ]
        elif side == "cooperative":
            camera_types = [
                "CAM_FRONT",
                "CAM_BACK",
                "CAM_LEFT",
                "CAM_RIGHT",
            ]
        elif side == "early-fusion":
            camera_types = [
                "CAM_FRONT",
                "CAM_BACK",
                "CAM_LEFT",
                "CAM_RIGHT",
                "CAM_FRONT_AIR",
                "CAM_BACK_AIR",
                "CAM_LEFT_AIR",
                "CAM_RIGHT_AIR",
                "CAM_BOTTOM_AIR",
            ]

        for cam in camera_types:
            if "AIR" in cam:
                cam_token = delay_sample["data"][cam]
            else:
                cam_token = sample["data"][cam]
            sample_data = nusc.get("sample_data", cam_token)
            calib_token = sample_data["calibrated_sensor_token"]
            calib_data = nusc.get("calibrated_sensor", calib_token)
            cam_info = {
                "data_path": sample_data["filename"],
                "type": sample_data["sensor_modality"],
                "sample_data_token": cam_token,
                "cam_intrinsic": calib_data["camera_intrinsic"],
                # For v2x-seq begin
                "lidar2cam_translation": calib_data["lidar2cam_translation"],
                "lidar2cam_rotation": calib_data["lidar2cam_rotation"],
                "sensor2lidar_translation": calib_data["sensor2lidar_translation"],
                "sensor2lidar_rotation": calib_data["sensor2lidar_rotation"],
                # For v2x-seq end
                "sensor2ego_translation": calib_data["translation"],
                "sensor2ego_rotation": calib_data["rotation"],
                "timestamp": sample_data["timestamp"],
            }
            info["cams"].update({cam: cam_info})

        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            # get_boxes() return boxes in anno coordinate, different from get_sample_data(), which is in sensor coordinate
            # boxes = nusc.get_boxes(lidar_token)
            _, boxes, _ = nusc.get_sample_data(lidar_token)  # lidar coordinate

            # UniAD Box
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            gt_boxes = np.concatenate(
                [locs, dims, -rots - np.pi / 2], axis=1
            )  # cx, cy, cz, w, l, h, yaw

            # # Sparse 4D Box
            # locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            # dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
            #     -1, 1
            # )
            # # convert box size to the format of our lidar coordinate system, which is x_size, y_size, z_size (corresponding to l, w, h)
            # gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"

            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )  # TODO: Check why in global coordinate
            valid_flag = np.ones(len(boxes), dtype=np.bool_)
            names = np.array([b.name for b in boxes])

            info["instance_inds"] = np.array(
                [nusc.getind("instance", x["instance_token"]) for x in annotations]
            )
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag

            # For v2x-seq begin
            info['gt_ins_tokens'] = np.array([b["instance_token"] for b in annotations])
            info['gt_inds'] = np.array([int(b["id"]) for b in annotations])
            info['anno_tokens'] = np.array([b["token"] for b in annotations])
            info['visibility_tokens'] = np.array(
                [b["visibility_token"] for b in annotations]
            )
            info['prev_anno_tokens'] = np.array([b["prev"] for b in annotations])
            info['next_anno_tokens'] = np.array([b["next"] for b in annotations])

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        if sample["scene_token"] in val_scenes:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="nuscenes converter")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./datasets/griffin_50scenes_25m/griffin-nuscenes",
    )
    parser.add_argument(
        "--out_path", type=str, default="./data/infos/griffin_50scenes_25m"
    )
    parser.add_argument("--info_prefix", type=str, default="griffin")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument(
        "--split_file",
        type=str,
        default="data/split_datas/griffin_50scenes_25m.json",
    )
    parser.add_argument(
        "--early_fusion",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--delay",
        action="store_true",
        help="whether to generate info file with 200, 400ms latency",
    )
    args = parser.parse_args()

    with open(args.split_file, 'r') as f:
        split_data = json.load(f)
    split_info = split_data['batch_split']

    print("Generating drone side info file...")
    create_nuscenes_infos(
        os.path.join(args.root_path, "drone-side"),
        os.path.join(args.out_path, "drone-side"),
        args.info_prefix,
        args.version,
        side="drone",
        split_info=split_info,
    )

    print("Generating vehicle side info file...")
    create_nuscenes_infos(
        os.path.join(args.root_path, "vehicle-side"),
        os.path.join(args.out_path, "vehicle-side"),
        args.info_prefix,
        args.version,
        side="vehicle",
        split_info=split_info,
    )

    print("Generating cooperative info file...")
    create_nuscenes_infos(
        os.path.join(args.root_path, "cooperative"),
        os.path.join(args.out_path, "cooperative"),
        args.info_prefix,
        args.version,
        side="cooperative",
        split_info=split_info,
    )

    if args.early_fusion:
        print("Generating early fusion info file...")
        create_nuscenes_infos(
            os.path.join(args.root_path, "early-fusion"),
            os.path.join(args.out_path, "early-fusion"),
            args.info_prefix,
            args.version,
            side="early-fusion",
            split_info=split_info,
        )

    if args.delay:
        # delay 200ms and 400ms
        for delay_frame_num in [2, 4]:
            print(
                f"Generating delay {delay_frame_num} frames info file for cooperative..."
            )
            create_nuscenes_infos(
                os.path.join(args.root_path, "cooperative"),
                os.path.join(args.out_path, f"cooperative-delay-{delay_frame_num}"),
                args.info_prefix,
                args.version,
                side="cooperative",
                split_info=split_info,
                delay_frame_num=delay_frame_num,
            )

            if args.early_fusion:
                print("Generating delay frame info file for early fusion...")
                create_nuscenes_infos(
                    os.path.join(args.root_path, "early-fusion"),
                    os.path.join(
                        args.out_path, f"early-fusion-delay-{delay_frame_num}"
                    ),
                    args.info_prefix,
                    args.version,
                    side="early-fusion",
                    split_info=split_info,
                    delay_frame_num=delay_frame_num,
                )

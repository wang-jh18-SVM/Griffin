import copy
import numpy as np
import torch
import mmcv
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from os import path as osp
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .eval_utils.nuscenes_eval import NuScenesEval_custom, TrackingEval_custom
from nuscenes.eval.common.config import config_factory
import tempfile
from mmcv.parallel import DataContainer as DC
import random
import pickle

from nuscenes import NuScenes
from .data_utils.data_utils import (
    lidar_nusc_box_to_global,
    output_to_nusc_box,
    output_to_nusc_box_det,
)
from nuscenes.prediction import convert_local_coords_to_global
import wandb


class_names_nuscenes_mappings = {
    'Car': 'car',
    'Truck': 'car',
    'Van': 'car',
    'Bus': 'car',
    'Motorcyclist': 'bicycle',
    'Cyclist': 'bicycle',
    'Tricyclist': 'bicycle',
    'Barrowlist': 'bicycle',
    'Pedestrian': 'pedestrian',
    'TrafficCone': 'traffic_cone',
    'car': 'car',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone'
}

@DATASETS.register_module()
class GriffinDataset(NuScenesDataset):
    """NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(
        self,
        inf_keys=[],
        queue_length=4,
        bev_size=(200, 200),
        patch_size=(102.4, 102.4),
        canvas_size=(200, 200),
        overlap_test=False,
        eval_mod=None,
        # For debug
        is_debug=False,
        len_debug=30,
        file_client_args=dict(backend='disk'),
        split_datas_file="",
        v2x_side='',
        class_range=None,
        new_range_100=False,
        enbale_temporal_aug=False,
        num_each_seq=20,
        seq_mode=False,
        *args,
        **kwargs,
    ):
        # init before super init since it is called in parent class
        self.split_datas_file = split_datas_file
        self.v2x_side = v2x_side
        if self.v2x_side not in [
            'vehicle-side',
            'infrastructure-side',
            'drone-side',
            'cooperative',
            'early-fusion',
        ]:
            raise Exception('v2x_side is not correct with {}'.format(self.v2x_side))
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)

        self.is_debug = is_debug
        self.len_debug = len_debug
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.scene_token = None
        self.eval_mod = eval_mod

        self.nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=True
        )

        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False

        self.class_range = class_range
        self.new_range_100 = new_range_100

        self.seq_mode = seq_mode
        if seq_mode:
            self.queue_length = 1
            self.num_each_seq = num_each_seq
            self._set_sequence_group_flag()
        self.record = []  # for clean data infos
        self.inf_keys = inf_keys

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        if self.num_each_seq == 0:
            curr_sequence = 0
            for idx in range(len(self.data_infos)):
                if idx != 0 and self.data_infos[idx]['frame_idx'] == 0:
                    # Not first frame and frame_idx=0 -> new sequence
                    curr_sequence += 1
                res.append(curr_sequence)
        else:
            curr_sequence = 0
            curr_num = 0
            for idx in range(len(self.data_infos)):
                if curr_num == self.num_each_seq:
                    curr_num = 1
                    curr_sequence += 1
                else:
                    curr_num += 1
                res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

    def __len__(self):
        if not self.is_debug:
            return len(self.data_infos)
        else:
            return self.len_debug

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args['backend'] == 'disk':
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[:: self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        elif self.file_client_args['backend'] == 'petrel':
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
            data_infos = data_infos[:: self.load_interval]
            self.metadata = data['metadata']
            self.version = self.metadata['version']
        else:
            assert False, 'Invalid file_client_args!'
        return data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_globals_3d: gt_globals of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame (list)
        """
        data_queue = []
        self.enbale_temporal_aug = False
        if self.enbale_temporal_aug:
            # temporal aug
            prev_indexs_list = list(range(index - self.queue_length, index))
            random.shuffle(prev_indexs_list)
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
            input_dict = self.get_data_info(index)
        else:
            # ensure the first and final frame in same scene
            final_index = index
            first_index = index - self.queue_length + 1
            if first_index < 0:
                return None
            if (
                self.data_infos[first_index]['scene_token']
                != self.data_infos[final_index]['scene_token']
            ):
                return None
            # current timestamp
            input_dict = self.get_data_info(final_index)
            prev_indexs_list = list(reversed(range(first_index, final_index)))
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        input_dict['index'] = index
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (
            example is None or ~(example['gt_labels_3d']._data != -1).any()
        ):
            if input_dict['sample_idx'] not in self.record:
                self.record.append(input_dict['sample_idx'])
                print('-------- Note Here --------')
                print('New Empty GT:', input_dict['sample_idx'], "with index:", index)
            else:
                print('-------- Note Here --------')
                print('Already in Record')

            for idx in self.record:
                print('Empty GT:', idx)
            return None
        data_queue.insert(0, example)

        # retrieve previous infos

        for i in prev_indexs_list:
            if self.enbale_temporal_aug:
                i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if (
                input_dict['frame_idx'] < frame_idx
                and input_dict['scene_token'] == scene_token
            ):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and (
                    example is None or ~(example['gt_labels_3d']._data != -1).any()
                ):
                    return None
                frame_idx = input_dict['frame_idx']

            data_queue.insert(0, copy.deepcopy(example))

        data_queue = self.union2one(data_queue)

        return data_queue

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """

        input_dict = self.get_data_info(index)
        input_dict['index'] = index
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        img_metas = example['img_metas']
        img_metas = img_metas[0].data

        example['img_metas'] = [DC(img_metas, cpu_only=True)]

        data_dict = {}
        for key, value in example.items():
            if 'l2g' in key or 'veh2inf' in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value
        return data_dict

    def union2one(self, queue):
        """
        convert sample dict into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
        l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
        timestamp_list = [
            to_tensor(np.array(each['timestamp'], dtype=np.float64)) for each in queue
        ]

        if self.v2x_side == 'cooperative':
            for key in self.inf_keys:
                queue[-1][key] = DC([each[key].data for each in queue])

        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if self.seq_mode:
                continue
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        queue['gt_labels_3d'] = DC(gt_labels_3d_list)
        queue['gt_inds'] = DC(gt_inds_list)
        queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
        queue['l2g_r_mat'] = DC(l2g_r_mat_list)
        queue['l2g_t'] = DC(l2g_t_list)
        queue['timestamp'] = DC(timestamp_list)
        return queue

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        print('Go into random idx!')
        pool = np.where(self.flag == self.flag[idx])[0]

        # Find the current index in the pool
        current_index_in_pool = np.where(pool == idx)[0]

        # Check if there is a next index in the pool
        if len(current_index_in_pool) > 0 and current_index_in_pool[0] + 1 < len(pool):
            # Select the next index
            next_idx = pool[current_index_in_pool[0] + 1]
        else:
            # If the current index is the last one or not found, return the first index in the pool
            next_idx = pool[0]

        if self.seq_mode:
            return next_idx
        return np.random.choice(pool)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - gt_inds (np.ndarray): Instance ids of ground truths.
                - gt_fut_traj (np.ndarray): .
                - gt_fut_traj_mask (np.ndarray): .
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]  # cx, cy, cz, w, l, h, yaw

        # Debug yaw
        gt_bboxes_3d[:, -1] = -gt_bboxes_3d[:, -1] + np.pi

        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_inds'][mask]

        sample = self.nusc.get('sample', info['token'])
        ann_tokens = np.array(sample['anns'])[mask]
        assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
        )
        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            # pts_filename=info['lidar_path'],
            # sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            # can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # Warning: For Right Multiplication Version. Normal Left Multiplication is (l2e_r_mat.T @ e2g_r_mat.T).T or e2g_r_mat @ l2e_r_mat
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(l2g_r_mat=l2g_r_mat.astype(np.float32), l2g_t=l2g_t.astype(np.float32))
        )

        if self.v2x_side == "cooperative":
            input_dict.update(
                dict(
                    veh2inf_rt=np.array(
                        info['vehLidar2airLidar_rt']
                    ).T,  # Warning: For Right Multiplication Version, otherwise use transpose
                )
            )
        if "air_sample_token" in info:
            input_dict.update(dict(sample_idx_inf=info['air_sample_token']))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                # #! Only use bottom camera for debug
                # if cam_type != 'CAM_BOTTOM':
                #     continue

                image_paths.append(cam_info['data_path'])

                # obtain lidar to image transformation matrix
                lidar2cam_r = Quaternion(cam_info['lidar2cam_rotation']).rotation_matrix
                lidar2cam_t = cam_info['lidar2cam_translation']
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r
                lidar2cam_rt[:3, 3] = lidar2cam_t

                intrinsic = np.array(cam_info['cam_intrinsic'])
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = np.zeros(18)
        can_bus[:3] = translation  # be set to zero in union2one
        can_bus[3:7] = rotation
        can_bus[7:10] = translation  # remain the same all along
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """

        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        nusc_map_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            if 'map' in self.eval_mod:
                map_annos = {}
                for key, value in det['ret_iou'].items():
                    map_annos[key] = float(value.numpy()[0])
                    nusc_map_annos[sample_token] = map_annos

            if 'boxes_3d' not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.class_range,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                # center_ = box.center.tolist()
                # change from ground height to center height
                # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                if name not in [
                    'car',
                    'truck',
                    'bus',
                    'trailer',
                    'motorcycle',
                    'bicycle',
                    'pedestrian',
                ]:
                    continue

                box_ego = boxes_ego[keep_idx[i]]
                trans = box_ego.center
                if 'traj' in det:
                    traj_local = det['traj'][keep_idx[i]].numpy()[..., :2]
                    traj_scores = det['traj_scores'][keep_idx[i]].numpy()
                else:
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))
                traj_ego = np.zeros_like(traj_local)
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi / 2)
                for kk in range(traj_ego.shape[0]):
                    traj_ego[kk] = convert_local_coords_to_global(
                        traj_local[kk], trans, rot
                    )

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    predict_traj=traj_ego,
                    predict_traj_score=traj_scores,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
            'map_results': nusc_map_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(
            self
        ), 'The length of results is not equal to the dataset len: {} != {}'.format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def _format_bbox_det(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]['token']

            if 'boxes_3d' not in det and 'boxes_3d_det' not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box_det(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.class_range,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc_det.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results_det(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(
            self
        ), 'The length of results is not equal to the dataset len: {} != {}'.format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results_det')
        else:
            tmp_dir = None

        result_files = self._format_bbox_det(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['pts_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if isinstance(results, dict):
            results = results['bbox_results']
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_files_det, tmp_dir = self.format_results_det(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], result_files_det[name]
                )
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, result_files_det)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def create_splits_griffin(self):
        split_data = mmcv.load(self.split_datas_file)

        return split_data['batch_split']

    def category_to_detection_name(self, category_name: str):
        if category_name in self.CLASSES:
            return category_name
        else:
            return None

    def category_to_tracking_name(self, category_name: str):
        if category_name in self.CLASSES:
            return category_name
        else:
            return None

    def _evaluate_single(
        self,
        result_path,
        result_path_det,
        logger=None,
        metric='bbox',
        result_name='pts_bbox',
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        output_dir_det = osp.join(output_dir, 'det')
        output_dir_track = osp.join(output_dir, 'track')
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',  #'train'
        }
        detail = dict()

        splits = self.create_splits_griffin()

        # category_to_type_name = self.category_to_detection_name
        # modify eval config according to config file.
        self.eval_detection_configs.class_names = self.CLASSES
        self.eval_detection_configs.class_range = self.class_range
        assert set(self.class_range.keys()) == set(self.CLASSES)
        if 'det' in self.eval_mod:
            try:
                self.nusc_eval = NuScenesEval_custom(
                    self.nusc,
                    config=self.eval_detection_configs,
                    result_path=result_path_det,
                    eval_set=eval_set_map[self.version],
                    output_dir=output_dir_det,
                    verbose=True,
                    overlap_test=self.overlap_test,
                    data_infos=self.data_infos,
                    splits=splits,
                    category_to_type_name=self.category_to_detection_name,
                    # class_range=self.class_range
                )
                self.nusc_eval.main(plot_examples=0, render_curves=False)
                # record metrics
                metrics = mmcv.load(osp.join(output_dir_det, 'metrics_summary.json'))
                metric_prefix = f'{result_name}'
                for name in self.CLASSES:
                    for k, v in metrics['label_aps'][name].items():
                        val = float('{:.4f}'.format(v))
                        detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
                    for k, v in metrics['label_tp_errors'][name].items():
                        val = float('{:.4f}'.format(v))
                        detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                    for k, v in metrics['tp_errors'].items():
                        val = float('{:.4f}'.format(v))
                        detail[
                            '{}/{}'.format(metric_prefix, self.ErrNameMapping[k])
                        ] = val
                detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
                detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
            except Exception as e:
                print(f"Error in detection evaluation: {e}")

        if 'track' in self.eval_mod:
            try:
                cfg = config_factory("tracking_nips_2019")
                cfg.class_names = self.CLASSES
                cfg.tracking_names = self.CLASSES
                cfg.class_range = self.class_range
                self.nusc_eval_track = TrackingEval_custom(
                    config=cfg,
                    result_path=result_path,
                    eval_set=eval_set_map[self.version],
                    output_dir=output_dir_track,
                    verbose=True,
                    nusc_version=self.version,
                    nusc_dataroot=self.data_root,
                    splits=splits,
                    category_to_type_name=self.category_to_tracking_name,
                    # class_range=self.class_range
                )
                self.nusc_eval_track.main(render_curves=False)
                # record metrics
                metrics = mmcv.load(osp.join(output_dir_track, 'metrics_summary.json'))
                keys = [
                    'amota',
                    'amotp',
                    'recall',
                    'motar',
                    'gt',
                    'mota',
                    'motp',
                    'mt',
                    'ml',
                    'faf',
                    'tp',
                    'fp',
                    'fn',
                    'ids',
                    'frag',
                    'tid',
                    'lgd',
                ]
                for key in keys:
                    detail['{}/{}'.format(metric_prefix, key)] = metrics[key]
            except Exception as e:
                print(f"Error in tracking evaluation: {e}")
        # Log metrics to wandb, begin with val/
        if wandb.run is not None:
            wandb.log(detail)

        return detail

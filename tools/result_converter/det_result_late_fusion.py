import argparse
import torch
import mmcv
import os
import warnings
from mmcv import Config
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from scipy.linalg import polar


def diff_label_filt(frame1, frame2, i, j):
    size = frame1.size[i]
    diff = np.abs(frame1.center[i] - frame2.center[j]) / size
    return (
        diff[0] <= 1
        and diff[1] <= 1
        and diff[2] <= 1
        and frame1.label[i] == frame2.label[j]
    )


class Bbox:
    def __init__(
        self,
        result_dict,
        inf2veh_r=None,
        inf2veh_t=None,
        post_center_range=None,
        score_thre=0.1,
    ) -> None:
        bboxes = result_dict['boxes_3d']
        scores = result_dict['scores_3d']
        labels = result_dict['labels_3d']
        if inf2veh_r is not None:
            assert post_center_range is not None
            score_mask = scores > score_thre
            bboxes = bboxes[score_mask]
            scores = scores[score_mask]
            labels = labels[score_mask]
            # self.trans_num = scores.shape[0]

            bboxes.rotate(inf2veh_r.T)
            bboxes.translate(inf2veh_t)

            mask = bboxes.in_range_3d(post_center_range)
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]

        # (n, 9): cx, cy, cz, w, l, h, rot, vx, vy
        self.boxes = bboxes.tensor.numpy()  # numpy() is shallow copy
        self.confidence = scores.numpy()
        self.label = labels.numpy()
        self.center = np.copy(self.boxes[:, :3])  # deep copy
        self.size = np.copy(self.boxes[:, 3:6])
        self.num_boxes = self.boxes.shape[0]
        self.num_dims = self.boxes.shape[1]
        assert self.num_boxes == self.confidence.shape[0] == self.label.shape[0]


class EuclidianMatcher:
    def __init__(self, filter_func=None, delta_x=0.0, delta_y=0.0, delta_z=0.0):
        super(EuclidianMatcher, self).__init__()
        self.filter_func = filter_func
        self.delta = [delta_x, delta_y, delta_z]

    def match(self, frame1, frame2):
        cost_matrix = np.zeros((frame1.num_boxes, frame2.num_boxes))
        for i in range(frame1.num_boxes):
            for j in range(frame2.num_boxes):
                cost_matrix[i][j] = (
                    np.sum((frame1.center[i] + self.delta - frame2.center[j]) ** 2)
                    ** 0.5
                )
                if self.filter_func is not None and not self.filter_func(
                    frame1, frame2, i, j
                ):
                    cost_matrix[i][j] = 1e6
        # print(cost_matrix, linear_sum_assignment(cost_matrix))
        index1, index2 = linear_sum_assignment(cost_matrix)
        accepted = []
        cost = 0
        for i in range(len(index1)):
            if cost_matrix[index1[i]][index2[i]] < 1e5:
                accepted.append(i)
                cost += cost_matrix[index1[i]][index2[i]]
        return (
            index1[accepted],
            index2[accepted],
            0 if len(accepted) == 0 else cost / len(accepted),
        )


class BasicFuser(object):
    def __init__(self, perspective, trust_type, retain_type):
        # perspective:
        # infrastructure / vehicle
        # trust type:
        # lc (Linear Combination) / max
        # retain type:
        # all / main / none
        self.perspective = perspective
        self.trust_type = trust_type
        self.retain_type = retain_type

    def fuse(self, frame_r, frame_v, ind_r, ind_v):
        if self.perspective == "infrastructure":
            frame1 = frame_r
            frame2 = frame_v
            ind1 = ind_r
            ind2 = ind_v
        elif self.perspective == "vehicle":
            frame1 = frame_v
            frame2 = frame_r
            ind1 = ind_v
            ind2 = ind_r

        confidence1 = np.array(frame1.confidence[ind1])
        confidence2 = np.array(frame2.confidence[ind2])
        if self.trust_type == "max":
            confidence1 = confidence1 > confidence2
            confidence2 = 1 - confidence1
        elif self.trust_type == "main":
            confidence1 = np.ones_like(confidence1)
            confidence2 = 1 - confidence1

        # (n, 3)
        center = frame1.center[ind1] * np.repeat(
            confidence1[:, np.newaxis], 3, axis=1
        ) + frame2.center[ind2] * np.repeat(confidence2[:, np.newaxis], 3, axis=1)
        boxes = np.copy(frame1.boxes[ind1])
        boxes[:, :3] += center
        boxes[:, :3] -= frame1.center[ind1]

        size = frame1.size[ind1] * np.repeat(
            confidence1[:, np.newaxis], 3, axis=1
        ) + frame2.size[ind2] * np.repeat(confidence2[:, np.newaxis], 3, axis=1)
        boxes[:, 3:6] += size
        boxes[:, 3:6] -= frame1.size[ind1]
        # boxes = (
        #     frame1.boxes[ind1][:, :3]
        #     + center
        #     - frame1.center[ind1]
        # )
        label = frame1.label[ind1]
        confidence = (
            frame1.confidence[ind1] * confidence1
            + frame2.confidence[ind2] * confidence2
        )
        # arrows = frame1.arrows[ind1]

        boxes_u = []
        label_u = []
        confidence_u = []
        # arrows_u = []
        if self.retain_type in ["all", "main"]:
            for i in range(frame1.num_boxes):
                if i not in ind1 and frame1.label[i] != -1:
                    boxes_u.append(frame1.boxes[i])
                    label_u.append(frame1.label[i])
                    confidence_u.append(frame1.confidence[i])
                    # arrows_u.append(frame1.arrows[i])

        if self.retain_type in ["all"]:
            for i in range(frame2.num_boxes):
                if i not in ind2 and frame2.label[i] != -1:
                    boxes_u.append(frame2.boxes[i])
                    label_u.append(frame2.label[i])
                    confidence_u.append(frame2.confidence[i])
                    # arrows_u.append(frame2.arrows[i])
        if len(boxes_u) == 0:
            result_dict = {
                "boxes_3d": boxes,
                # "arrows": arrows,
                "labels_3d": label,
                "scores_3d": confidence,
            }
        else:
            result_dict = {
                "boxes_3d": np.concatenate((boxes, np.array(boxes_u)), axis=0),
                # "arrows": np.concatenate((arrows, np.array(arrows_u)), axis=0),
                "labels_3d": np.concatenate((label, np.array(label_u)), axis=0),
                "scores_3d": np.concatenate(
                    (confidence, np.array(confidence_u)), axis=0
                ),
            }
        return result_dict


class LateFusion(nn.Module):
    def __init__(self, dataset_type, post_center_range):
        super(LateFusion, self).__init__()
        self.dataset_type = dataset_type
        self.post_center_range = post_center_range
        self.matcher = EuclidianMatcher(diff_label_filt)
        self.fuser = BasicFuser(
            perspective="vehicle", trust_type="main", retain_type="all"
        )
        # self.num = 0

    def forward(self, veh_res, inf_res, veh2inf_rt):
        inf2veh_r, inf2veh_t = self.inverse_rt(veh2inf_rt)
        pred_veh = Bbox(veh_res)
        pred_inf = Bbox(inf_res, inf2veh_r, inf2veh_t, self.post_center_range)
        # self.num += pred_inf.trans_num
        # print('\n', self.num)

        ind_inf, ind_veh, cost = self.matcher.match(pred_inf, pred_veh)
        fuse_result = self.fuser.fuse(pred_inf, pred_veh, ind_inf, ind_veh)
        for k, v in fuse_result.items():
            fuse_result[k] = torch.from_numpy(v)

        res_dict = {
            'boxes_3d': LiDARInstance3DBoxes(
                fuse_result['boxes_3d'], box_dim=fuse_result['boxes_3d'].shape[1]
            ),
            'labels_3d': fuse_result['labels_3d'],
            'scores_3d': fuse_result['scores_3d'],
        }
        return res_dict

    def iterative_closest_point(self, A, num_iterations=100):
        R = A.copy()
        for _ in range(num_iterations):
            U, _ = polar(R)
            R = U
        return R

    def inverse_rt(self, rt):
        rt = rt.cpu()[0].T
        rot = rt[:3, :3].numpy()
        trans = rt[:3, 3].numpy()
        if "spd" in self.dataset_type:
            appro_rot = self.iterative_closest_point(rot)
            inv_rot = np.linalg.inv(appro_rot)
        else:
            inv_rot = np.linalg.inv(rot)
        inv_trans = -np.dot(inv_rot, trans)

        return inv_rot, inv_trans


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument(
        '--config',
        type=str,
        default='projects/configs_griffin_50scenes_25m/cooperative/tiny_track_r50_stream_bs1_3cls_late_fusion.py',
        help='test config file path',
    )
    parser.add_argument(
        '--veh-det-pkl',
        type=str,
        default='projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211032.pkl',
        help='Path of detection results from vehicle-side',
    )
    parser.add_argument(
        '--inf-det-pkl',
        type=str,
        default='projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_48epoch_3cls/results-02211258.pkl',
        help='Path of detection results from air-side',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output result file in pickle format',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode, set inf-det-pkl to the same as veh-det-pkl, expected same metrics as single-side detection',
    )
    args = parser.parse_args()
    args.eval = 'bbox'

    return args


def main():
    args = parse_args()

    print(f'Loading vehicle-side detection results from {args.veh_det_pkl}')
    veh_det_pkl = mmcv.load(args.veh_det_pkl)
    veh_res = veh_det_pkl['bbox_results']
    veh_res_dict = {}
    for v_res in veh_res:
        veh_res_dict[v_res['token']] = v_res
    print(f'Loading infrastructure-side detection results from {args.inf_det_pkl}')
    inf_det_pkl = mmcv.load(args.inf_det_pkl)
    inf_res = inf_det_pkl['bbox_results']
    inf_res_dict = {}
    for i_res in inf_res:
        inf_res_dict[i_res['token']] = i_res

    assert len(veh_res) == len(
        inf_res
    ), f'Number of vehicle-side results ({len(veh_res)}) does not match number of infrastructure-side results ({len(inf_res)})'

    # for v_res, i_res in zip(veh_res, inf_res):
    #     assert (
    #         v_res['token'] == i_res['token']
    #     ), f'Vehicle token ({v_res["token"]}) does not match infrastructure token ({i_res["token"]})'

    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f'{current_folder_path}/../..')
    curDirectory = os.getcwd()
    print(curDirectory)

    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # # import modules from string list.
    # if cfg.get('custom_imports', None):
    #     from mmcv.utils import import_modules_from_strings

    #     import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    distributed = False

    # build the dataloader
    if args.debug:
        cfg.data.test.is_debug = True
        cfg.data.test.len_debug = 30
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # When debug is True, or use delayed dataset info pkl, the number of vehicle-side results may be less than the number of dataset
    assert len(veh_res) >= len(
        dataset
    ), f'Number of vehicle-side results ({len(veh_res)}) does not match number of dataset ({len(dataset)})'

    late_fusion_model = LateFusion(
        cfg.data.test.type.lower(), cfg.model.post_center_range
    )
    outputs = {
        'bbox_results': [],
    }
    print(f'Processing late fusion for {len(data_loader)} samples')
    for info in mmcv.track_iter_progress(data_loader):
        img_metas = info['img_metas'][0].data[0][0]
        sample_idx_veh = img_metas['sample_idx']
        sample_idx_inf = (
            img_metas['sample_idx_inf']
            if 'sample_idx_inf' in img_metas
            else sample_idx_veh
        )
        assert (
            sample_idx_veh in veh_res_dict
        ), f'Sample index ({sample_idx_veh}) not found in vehicle-side results'
        assert (
            sample_idx_inf in inf_res_dict
        ), f'Sample index ({sample_idx_inf}) not found in infrastructure-side results'

        veh2inf_rt = info['veh2inf_rt']
        veh_result = veh_res_dict[sample_idx_veh]
        inf_result = inf_res_dict[sample_idx_inf]

        if args.debug:
            veh2inf_rt = torch.from_numpy(np.eye(4)).unsqueeze(0)
            inf_result = veh_result

        late_fusion_result = late_fusion_model(veh_result, inf_result, veh2inf_rt)
        outputs['bbox_results'].append(
            {
                "token": sample_idx_veh,
                "boxes_3d": late_fusion_result['boxes_3d'],
                "scores_3d": late_fusion_result['scores_3d'],
                "labels_3d": late_fusion_result['labels_3d'],
            }
        )

    if args.out:
        mmcv.dump(outputs, args.out)

    kwargs = {}
    kwargs['jsonfile_prefix'] = osp.join(
        args.config.replace('projects/configs', 'projects/work_dirs').split('.')[0],
        time.ctime().replace(' ', '_').replace(':', '_'),
    )
    # if args.format_only:
    #     dataset.format_results(outputs, **kwargs)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
        'interval',
        'tmpdir',
        'start',
        'gpu_collect',
        'save_best',
        'rule',
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))

    print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()

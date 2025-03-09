import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from .agile_track import AgileTrack


@DETECTORS.register_module()
class Agile(AgileTrack):
    def __init__(
        self,
        seq_mode=False,
        batch_size=1,
        is_cooperation=False,
        drop_rate=0,
        save_track_query=False,
        save_track_query_file_root='',
        read_track_query_file_root='',
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        inf_pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        **kwargs,
    ):
        super(Agile, self).__init__(
            is_cooperation=is_cooperation,
            seq_mode=seq_mode,
            batch_size=batch_size,
            pc_range=pc_range,
            inf_pc_range=inf_pc_range,
            post_center_range=post_center_range,
            drop_rate=drop_rate,
            **kwargs,
        )

        self.task_loss_weight = {"track": 1.0}

        self.is_cooperation = is_cooperation
        # self.drop_rate = drop_rate
        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root
        self.read_track_query_file_root = read_track_query_file_root

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        # for coop
        veh2inf_rt=None,
        **kwargs,  # [1, 9]
    ):
        """Forward training function for the model.

        Args:
        img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
        img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
        gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
        gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
        gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
        l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
        l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
        timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
        gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
        gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
        gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
        gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
        veh2inf_rt (list[torch.Tensor], optional): List of tensors containing vehicle-to-infrastructure transformation matrices. Defaults to None.

        Returns:
            dict: Dictionary containing losses.
        """
        if self.test_flag:  # for interval evaluation
            # import ipdb;ipdb.set_trace()
            self.reset_memory()
            self.test_flag = False
        # import pdb;pdb.set_trace()
        losses = dict()
        len_queue = img.size(1)  # [1, 1, 5, 3, 544, 960]
        if self.seq_mode:
            losses_track, outs_track = self.forward_track_stream_train(
                img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_inds,
                l2g_t,
                l2g_r_mat,
                img_metas,
                timestamp,
                veh2inf_rt,
                **kwargs,
            )
        else:
            losses_track, outs_track = self.forward_track_train(
                img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_inds,
                l2g_t,
                l2g_r_mat,
                img_metas,
                timestamp,
                veh2inf_rt,
            )
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)

        # Upsample bev for tiny version
        # outs_track = self.upsample_bev_if_tiny(outs_track)

        # bev_embed = outs_track["bev_embed"]
        # bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue - 1] for each in img_metas]

        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses

    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(
        self,
        img=None,
        img_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        # for coop
        veh2inf_rt=None,
        **kwargs,
    ):
        """Test function"""
        # import ipdb;ipdb.set_trace()
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        img = [img] if img is None else img

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(
            img, l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs
        )

        # Upsample bev for tiny model
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])

        bev_embed = result_track[0]["bev_embed"]

        pop_track_list = [
            'prev_bev',
            'bev_pos',
            'bev_embed',
            'track_query_embeddings',
            'sdc_embedding',
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])

        return result

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        from mmdet3d.core import (
            Box3DMode,
            Coord3DMode,
            bbox3d2result,
            merge_aug_bboxes_3d,
            show_result,
        )
        import mmcv
        from os import path as osp
        from mmcv.parallel import DataContainer as DC

        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(
                    f"Unsupported data type {type(data['points'][0])} "
                    f'for visualization!'
                )
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id]['box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!'
                )
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            # inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            # pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            inds = result[batch_id]['scores_3d'].numpy() > 0.1
            pred_bboxes = result[batch_id]['boxes_3d'][inds]

            if len(pred_bboxes) <= 0:
                continue

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(
                    points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
                )
                pred_bboxes = Box3DMode.convert(
                    pred_bboxes, box_mode_3d, Box3DMode.DEPTH
                )
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result

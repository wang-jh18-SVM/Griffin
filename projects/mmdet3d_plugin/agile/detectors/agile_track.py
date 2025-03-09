import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from ..dense_heads.track_head_plugin import (
    MemoryBank,
    QueryInteractionModule,
    Instances,
    RuntimeTrackerBase,
)
from ..modules import CrossAgentSparseInteraction
import mmcv, os
import numpy as np


@DETECTORS.register_module()
class AgileTrack(MVXTwoStageDetector):
    """Agile tracking part"""

    def __init__(
        self,
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        inf_pc_range=None,
        post_center_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
        is_cooperation=False,
        drop_rate=0,
        save_track_query=False,
        save_track_query_file_root='',
        seq_mode=False,
        batch_size=1,
    ):
        super(AgileTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        bbox_coder['pc_range'] = pc_range
        bbox_coder['post_center_range'] = post_center_range
        bbox_coder['num_classes'] = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.criterion = build_loss(loss_cfg)
        # for test memory
        self.scene_token = None
        self.timestamp = None
        self.prev_bev = None
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

        # cross-agent query interaction
        self.is_cooperation = is_cooperation
        if self.is_cooperation:
            self.cross_agent_query_interaction = CrossAgentSparseInteraction(
                pc_range=self.pc_range,
                inf_pc_range=self.inf_pc_range,
                embed_dims=self.embed_dims,
            )
            # self.cross_agent_query_interaction_coop = CrossAgentSparseInteractionCoop(pc_range=self.pc_range,
            #                                                                     inf_pc_range=self.inf_pc_range,
            #                                                                         embed_dims=self.embed_dims)
        self.drop_rate = drop_rate

        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root

        self.bev_embed_linear = nn.Linear(embed_dims, embed_dims)
        self.bev_pos_linear = nn.Linear(embed_dims, embed_dims)

        self.seq_mode = seq_mode
        if self.seq_mode:
            self.batch_size = batch_size
            self.test_flag = False
            # for stream train memory
            self.train_prev_infos = {
                'scene_token': [None] * self.batch_size,
                'prev_timestamp': [None] * self.batch_size,
                'prev_bev': [None] * self.batch_size,
                'track_instances': [None] * self.batch_size,
                'l2g_r_mat': [None] * self.batch_size,
                'l2g_t': [None] * self.batch_size,
                'prev_pos': [0] * self.batch_size,
                'prev_angle': [0] * self.batch_size,
            }

    def reset_memory(self):
        self.train_prev_infos['scene_token'] = [None] * self.batch_size
        self.train_prev_infos['prev_timestamp'] = [None] * self.batch_size
        self.train_prev_infos['prev_bev'] = [None] * self.batch_size
        self.train_prev_infos['track_instances'] = [None] * self.batch_size
        self.train_prev_infos['l2g_r_mat'] = [None] * self.batch_size
        self.train_prev_infos['l2g_t'] = [None] * self.batch_size
        self.train_prev_infos['prev_pos'] = [0] * self.batch_size
        self.train_prev_infos['prev_angle'] = [0] * self.batch_size

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B // len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        object_query = self.query_embedding.weight
        query_embeds, query_feats = torch.split(object_query, self.embed_dims, dim=1)
        track_instances.ref_pts = self.reference_points(query_embeds).sigmoid()

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query_feats = query_feats
        track_instances.query_embeds = query_embeds

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def _init_inf_tracks(self, inf_dict):
        track_instances = Instances((1, 1))
        device = inf_dict['ref_pts'].device
        num_queries = inf_dict['ref_pts'].shape[0]
        dim = self.embed_dims * 2
        track_instances.ref_pts = inf_dict['ref_pts']

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query_feats = inf_dict['query_feats']
        track_instances.query_embeds = inf_dict['query_embeds']

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = inf_dict['obj_idxes']
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        """
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        # reference_points = ref_pts.sigmoid().clone()
        reference_points = ref_pts.clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
        # g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)
        # for 4090, move to cpu, inverse and then move to cuda
        g2l_r = torch.linalg.inv(l2g_r2.to('cpu')).to(ref_pts.device).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )

        # ref_pts = inverse_sigmoid(ref_pts)
        # ref_pts = ref_pts.clamp(min=0, max=1)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
                )
        self.train()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(
        self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None
    ):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
                )
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
            )

        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    def _get_coop_bev_embed(
        self, bev_embed_src, bev_pos_src, track_instances, start_idx
    ):
        bev_embed = bev_embed_src
        bev_pos = bev_pos_src
        act_track_instances = track_instances[:start_idx]

        # print('act_track_instances len:',len(act_track_instances))

        locs = act_track_instances.ref_pts.clone()
        locs[:, 0:1] = locs[:, 0:1] * self.bev_w  # w
        locs[:, 1:2] = locs[:, 1:2] * self.bev_h  # h

        pixel_len = 1  # 2

        for idx in range(act_track_instances.ref_pts.shape[0]):
            w = int(locs[idx, 0])
            h = int(locs[idx, 1])
            if w >= self.bev_w or w < 0 or h >= self.bev_h or h < 0:
                continue
            # bev_embed: [bev_h * bev_w, bs, embed_dims],
            for hh in range(max(0, h - pixel_len), min(self.bev_h - 1, h + pixel_len)):
                for ww in range(
                    max(0, w - pixel_len), min(self.bev_w - 1, w + pixel_len)
                ):
                    bev_embed[hh * self.bev_w + ww, :, :] = bev_embed[
                        hh * self.bev_w + ww, :, :
                    ] + self.bev_embed_linear(act_track_instances.query_feats[idx])
                    bev_pos[:, :, hh, ww] = bev_pos[:, :, hh, ww] + self.bev_pos_linear(
                        act_track_instances.query_embeds[idx]
                    )

        return bev_embed, bev_pos

    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train_bs(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
        veh2inf_rt=None,
        prev_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        **kwargs
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # import ipdb;ipdb.set_trace()
        assert self.batch_size == len(track_instances)
        if self.seq_mode:  # for stream training
            for i in range(self.batch_size):
                if l2g_r1[i] is None:
                    # print('not have prev')
                    continue
                ref_pts = track_instances[i].ref_pts
                velo = track_instances[i].pred_boxes[:, -2:]
                ref_pts = self.velo_update(
                    ref_pts,
                    velo,
                    l2g_r1[i],
                    l2g_t1[i],
                    l2g_r2[i],
                    l2g_t2[i],
                    time_delta=time_delta[i],
                )
                ref_pts = ref_pts.squeeze(0)
                track_instances[i].ref_pts[..., :2] = ref_pts[..., :2]
                # print(track_instances[i].ref_pts._version)

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        # import pdb;pdb.set_trace()
        if not self.seq_mode:
            bev_embed, bev_pos = self.get_bevs(
                img,
                img_metas,
                prev_img=prev_img,
                prev_img_metas=prev_img_metas,
            )
        else:
            # [bev_h * bev_w, bs, channel]
            bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        # cross-agent query interaction
        if self.is_cooperation:
            for i in range(self.batch_size):
                inf_dict = {
                    'query_feats': kwargs['query_feats'][i][0],
                    'query_embeds': kwargs['query_embeds'][i][0],
                    'ref_pts': kwargs['ref_pts'][i][0],  # [num_query, 3], range: 0~1
                    'obj_idxes': kwargs['obj_idxes'][i][0].long(),
                }
                if inf_dict['query_feats'].shape[0] > 0:
                    inf_instances = self._init_inf_tracks(inf_dict)

                cur_track_instances = track_instances[i]
                cur_track_instances, num_inf = self.cross_agent_query_interaction(
                    inf_instances, cur_track_instances, veh2inf_rt[i]
                )
                # print('track_nums_src: %d, track_nums_new: %d'%(track_nums_src,track_nums_new))
                # bev_embed: [2500, 1, 256] bev_pos: [1, 256, 50, 50]
                # import pdb;pdb.set_trace()
                bev_embed[:, i : i + 1, :], bev_pos[i : i + 1, :, :, :] = (
                    self._get_coop_bev_embed(
                        bev_embed[:, i : i + 1, :],
                        bev_pos[i : i + 1, :, :, :],
                        cur_track_instances,
                        num_inf,
                    )
                )
                track_instances[i] = cur_track_instances

        # here ref_pts need to clone to avoid error
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
        # but I can't find the reason
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            query_feats=torch.stack([ins.query_feats for ins in track_instances]),
            query_embeds=torch.stack([ins.query_embeds for ins in track_instances]),
            ref_points=torch.stack([ins.ref_pts for ins in track_instances]),
            img_metas=img_metas,
        )

        output_classes = det_output[
            "all_cls_scores"
        ]  # [num_layers, bs, num_query, num_cls]
        output_coords = det_output[
            "all_bbox_preds"
        ]  # [num_layers, bs, num_query, num_dim] # lidar coord
        # output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]  # [bs, num_query, 3]
        query_feats = det_output[
            "query_feats"
        ]  # [num_layers, bs, num_query, embed_dims]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],  # lidar coord
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,  # [bev_h * bev_w, bs, channel]
            "bev_pos": bev_pos,
        }
        losses = {}
        avg_factors = {}

        for j in range(self.batch_size):
            # num_views = img.size(1)  #!for single camera per view
            num_frames = len(gt_bboxes_3d[j])

            # init gt instances!
            gt_instances_list = []
            for i in range(num_frames):
                gt_instances = Instances((1, 1))
                boxes = gt_bboxes_3d[j][i].tensor.to(img.device)  # lidar coord
                # normalize gt bboxes here! # x, y, w, l, z, h, rot.sin(), rot.cos(), vx, vy
                boxes = normalize_bbox(boxes, self.pc_range)
                gt_instances.boxes = boxes
                gt_instances.labels = gt_labels_3d[j][i]
                gt_instances.obj_ids = gt_inds[j][i]
                gt_instances_list.append(gt_instances)
            self.criterion.initialize_for_single_clip(gt_instances_list)

            single_out = {}
            with torch.no_grad():
                track_scores = output_classes[-1, j, :].sigmoid().max(dim=-1).values

            # Step-1 Update track instances with current prediction
            # [nb_dec, bs, num_query, xxx]
            nb_dec = output_classes.size(0)

            # the track id will be assigned by the matcher.
            track_instances_list = [
                self._copy_tracks_for_loss(track_instances[j])
                for i in range(nb_dec - 1)
            ]
            track_instances[j].output_embedding = query_feats[-1][j]  # [900, feat_dim]
            velo = output_coords[-1, j, :, -2:]  # [num_query, 3]
            # import ipdb;ipdb.set_trace()
            if l2g_r2[j] is not None and not self.seq_mode:  # for sliding window
                # Update ref_pts for next frame considering each agent's velocity
                ref_pts = self.velo_update(
                    last_ref_pts[j],
                    velo,
                    l2g_r1[j],
                    l2g_t1[j],
                    l2g_r2[j],
                    l2g_t2[j],
                    time_delta=time_delta[j],
                )
            else:
                ref_pts = last_ref_pts[j]

            # track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
            # track_instances[j].ref_pts[...,:2] = ref_pts[...,:2]
            track_instances[j].ref_pts = ref_pts  # lidar coord
            track_instances_list.append(track_instances[j])

            for i in range(nb_dec):
                track_instances_tmp = track_instances_list[i]

                track_instances_tmp.scores = track_scores
                track_instances_tmp.pred_logits = output_classes[i, j]  # [300, num_cls]
                track_instances_tmp.pred_boxes = output_coords[
                    i, j
                ]  # [300, box_dim] # lidar coord
                # track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

                single_out["track_instances"] = track_instances_tmp
                track_instances_tmp, matched_indices = (
                    self.criterion.match_for_single_frame(
                        single_out, i, if_step=(i == (nb_dec - 1))
                    )
                )
                all_query_embeddings[j].append(query_feats[i][j])
                all_matched_indices[j].append(matched_indices)
                all_instances_pred_logits[j].append(output_classes[i, j])
                all_instances_pred_boxes[j].append(output_coords[i, j])  # Not used

            active_index = (
                (track_instances_tmp.obj_idxes >= 0)
                & (track_instances_tmp.iou >= self.gt_iou_threshold)
                & (track_instances_tmp.matched_gt_idxes >= 0)
            )
            single_out.update(
                self.select_active_track_query(
                    track_instances_tmp, active_index, [img_metas[j]]
                )
            )
            # out.update(self.select_sdc_track_query(track_instances[900], img_metas))

            # memory bank
            if self.memory_bank is not None:
                track_instances_tmp = self.memory_bank(track_instances_tmp)
            # Step-2 Update track instances using matcher

            tmp = {}
            tmp["init_track_instances"] = self._generate_empty_tracks()
            tmp["track_instances"] = track_instances_tmp
            out_track_instances = self.query_interact(tmp, is_drop=True)
            single_out["track_instances"] = out_track_instances

            for key, value in single_out.items():
                if key not in out:
                    out[key] = []
                out[key].append(value)

            for key, value in self.criterion.losses_dict.items():
                if 'loss' not in key:
                    continue
                af_key = key.replace('loss', 'avg_factor')
                avg_factor = self.criterion.losses_dict[af_key]
                if key not in losses:
                    losses[key] = value * avg_factor
                    avg_factors[af_key] = avg_factor
                else:
                    new_value = losses[key] + value * avg_factor
                    losses[key] = new_value
                    new_avg_factor = avg_factors[af_key] + avg_factor
                    avg_factors[af_key] = new_avg_factor

        for key, value in losses.items():
            af_key = key.replace('loss', 'avg_factor')
            avg_factor = avg_factors[af_key]
            losses[key] = value / avg_factor

        return out, losses

    def select_active_track_query(
        self, track_instances, active_index, img_metas, with_mask=True
    ):
        result_dict = self._track_instances2results(
            track_instances[active_index], img_metas, with_mask=with_mask
        )
        result_dict["track_query_embeddings"] = track_instances.output_embedding[
            active_index
        ][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[
            active_index
        ][result_dict['bbox_index']][result_dict['mask']]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(
            sdc_instance, img_metas, with_mask=False
        )
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(
        self,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_inds,
        l2g_t,
        l2g_r_mat,
        img_metas,
        timestamp,
        veh2inf_rt,
    ):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        num_frame = img.size(1)
        # init gt instances!
        gt_instances_list = []

        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)

            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            # TODO: Generate prev_bev in an RNN way.

            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            # frame_res = self._forward_single_frame_train_coop(
            frame_res = self._forward_single_frame_train(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
                veh2inf_rt,
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]

        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_query_matched_idxes",
            "track_bbox_results",
        ]
        out.update({k: frame_res[k] for k in get_keys})

        losses = self.criterion.losses_dict
        return losses, out

    def forward_track_stream_train(
        self,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_inds,
        l2g_t,
        l2g_r_mat,
        img_metas,
        timestamp,
        veh2inf_rt,
        **kwargs
    ):
        """Forward funciton
        Args:
        Returns:
        """
        assert self.batch_size == img.size(0)
        # import ipdb;ipdb.set_trace()
        time_delta = [None] * self.batch_size
        l2g_r1 = [None] * self.batch_size
        l2g_t1 = [None] * self.batch_size
        l2g_r2 = [None] * self.batch_size
        l2g_t2 = [None] * self.batch_size
        for i in range(self.batch_size):
            tmp_pos = copy.deepcopy(img_metas[i][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[i][0]['can_bus'][-1])
            if (
                img_metas[i][0]['scene_token']
                != self.train_prev_infos['scene_token'][i]
                or timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i] > 0.5
                or timestamp[i][0] < self.train_prev_infos['prev_timestamp'][i]
            ):
                # print('change----------------')
                # print(img_metas[0][0]['scene_token'], self.train_prev_infos['scene_token'])
                # the first sample of each scene is truncated
                self.train_prev_infos['track_instances'][
                    i
                ] = self._generate_empty_tracks()
                # print('ref_pts:', id(self.train_prev_infos['track_instances'][i].ref_pts))
                # print('query:', id(self.train_prev_infos['track_instances'][i].query))
                self.train_prev_infos['prev_bev'][i] = None
                time_delta[i], l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i] = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                img_metas[i][0]['can_bus'][:3] = 0
                img_metas[i][0]['can_bus'][-1] = 0
            else:
                time_delta[i] = (
                    timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i]
                )
                # print(time_delta)
                assert time_delta[i] > 0
                l2g_r1[i] = self.train_prev_infos['l2g_r_mat'][i]
                l2g_t1[i] = self.train_prev_infos['l2g_t'][i]
                l2g_r2[i] = l2g_r_mat[i][0]
                l2g_t2[i] = l2g_t[i][0]
                img_metas[i][0]['can_bus'][:3] -= self.train_prev_infos['prev_pos'][i]
                img_metas[i][0]['can_bus'][-1] -= self.train_prev_infos['prev_angle'][i]

            # update prev_infos
            # timestamp[0][0]: the first 0 is batch, the second 0 is num_frame
            self.train_prev_infos['scene_token'][i] = img_metas[i][0]['scene_token']
            self.train_prev_infos['prev_timestamp'][i] = timestamp[i][0]
            self.train_prev_infos['l2g_r_mat'][i] = l2g_r_mat[i][0]
            self.train_prev_infos['l2g_t'][i] = l2g_t[i][0]
            self.train_prev_infos['prev_pos'][i] = tmp_pos
            self.train_prev_infos['prev_angle'][i] = tmp_angle

        prev_bev = torch.stack(
            [
                (
                    bev
                    if isinstance(bev, torch.Tensor)
                    else torch.zeros(
                        [
                            self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w,
                            self.pts_bbox_head.in_channels,
                        ]
                    ).to(img.device)
                )
                for bev in self.train_prev_infos['prev_bev']
            ]
        )
        # prev_bev = self.train_prev_infos['prev_bev']
        track_instances = self.train_prev_infos['track_instances']
        # track_instances = self._generate_empty_tracks()

        out = dict()
        # import ipdb;ipdb.set_trace()
        img_single = torch.stack([img_[0] for img_ in img], dim=0)
        img_metas_single = [
            copy.deepcopy(img_metas[i][0]) for i in range(self.batch_size)
        ]
        all_query_embeddings = [[]] * self.batch_size
        all_matched_idxes = [[]] * self.batch_size
        all_instances_pred_logits = [[]] * self.batch_size
        all_instances_pred_boxes = [[]] * self.batch_size
        frame_res, losses = self._forward_single_frame_train_bs(
            img_single,
            img_metas_single,
            track_instances,
            None,  # prev_img
            None,  # prev_img_metas
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            all_query_embeddings,
            all_matched_idxes,
            all_instances_pred_logits,
            all_instances_pred_boxes,
            veh2inf_rt,
            prev_bev=prev_bev,
            gt_bboxes_3d=gt_bboxes_3d,  # lidar coord
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            **kwargs
        )
        # all_query_embeddings: len=dec nums, N*256
        # all_matched_idxes: len=dec nums, N*2
        track_instances = frame_res["track_instances"]

        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_query_matched_idxes",
            "track_bbox_results",
        ]
        out.update({k: frame_res[k] for k in get_keys})
        bev_embed = out['bev_embed'].detach().clone()
        for i in range(self.batch_size):
            self.train_prev_infos['prev_bev'][i] = bev_embed[:, i, :]
            self.train_prev_infos['track_instances'][i] = track_instances[
                i
            ].detach_and_clone()

        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"]  # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(
                bev_embed, '(h w) b c -> b c h w', h=h, w=w
            )  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(
                        prev_bev
                    )  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(
                        prev_bev
                    )  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        **kwargs
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            # dim = active_inst.query.shape[-1]
            # active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[..., :2] = ref_pts[..., :2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        # cross-agent query interaction
        if self.is_cooperation:
            drop_rate = self.drop_rate
            nums = np.random.choice([0, 1], size=1, p=[drop_rate, 1 - drop_rate])
            if nums[0]:
                inf_dict = {
                    'query_feats': kwargs['query_feats'][0][0],
                    'query_embeds': kwargs['query_embeds'][0][0],
                    'ref_pts': kwargs['ref_pts'][0][0],
                    'obj_idxes': kwargs['obj_idxes'][0][0].long(),
                }
                if inf_dict['query_feats'].shape[0] > 0:
                    inf_instances = self._init_inf_tracks(inf_dict)
                track_instances, num_inf = self.cross_agent_query_interaction(
                    inf_instances, track_instances, veh2inf_rt[0]
                )
                # print('track_nums_src: %d, track_nums_new: %d'%(track_nums_src,track_nums_new))
                # bev_embed: [2500, 1, 256] bev_pos: [1, 256, 50, 50]
                # import pdb;pdb.set_trace()
                bev_embed, bev_pos = self._get_coop_bev_embed(
                    bev_embed, bev_pos, track_instances, num_inf
                )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            query_feats=track_instances.query_feats[None, :, :],
            query_embeds=track_instances.query_embeds[None, :, :],
            ref_points=track_instances.ref_pts[None, :, :],
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            # "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        """ update track base """
        self.track_base.update(track_instances, None)

        active_index = (track_instances.obj_idxes >= 0) & (
            track_instances.scores >= self.track_base.filter_score_thresh
        )  # filter out sleep objects
        out.update(
            self.select_active_track_query(track_instances, active_index, img_metas)
        )
        # out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        veh2inf_rt=None,
        **kwargs
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        assert bs == 1, "only support bs=1"
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
            or timestamp - self.timestamp > 0.5
            or timestamp < self.timestamp
        ):
            # print('------- scene change -------')
            # print(img_metas[0]["scene_token"], self.scene_token)
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            img_metas[0]['can_bus'][:3] = 0
            img_metas[0]['can_bus'][-1] = 0
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            # print(time_delta, img_metas[0]['can_bus'][:3], img_metas[0]['can_bus'][-1])
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        """ predict and update """
        prev_bev = self.prev_bev
        # frame_res = self._forward_single_frame_inference_coop(
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt,
            **kwargs
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        self.test_track_instances = track_instances

        results = [dict()]
        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_bbox_results",
            "boxes_3d",
            "scores_3d",
            "labels_3d",
            "track_scores",
            "track_ids",
        ]
        results[0].update({k: frame_res[k] for k in get_keys})

        ## UniV2X: inf_track_query
        if self.save_track_query:
            tensor_to_cpu = torch.zeros(1)
            save_path = os.path.join(
                self.save_track_query_file_root, img_metas[0]['sample_idx'] + '.pkl'
            )
            track_instances = track_instances.to(tensor_to_cpu)
            mmcv.dump(track_instances, save_path)

        results = self._det_instances2results(
            track_instances_fordet, results, img_metas
        )
        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
        bboxes_dict = self.bbox_coder.decode(
            bbox_dict, with_mask=with_mask, img_metas=img_metas
        )[0]
        bboxes = bboxes_dict["bboxes"]  # [N, 9]:x, y, z, w, l, h, rot, vx, vy
        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[
                [
                    bboxes.to("cpu"),
                    scores.cpu(),
                    labels.cpu(),
                    bbox_index.cpu(),
                    bboxes_dict["mask"].cpu(),
                ]
            ],
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]

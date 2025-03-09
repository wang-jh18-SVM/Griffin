import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..dense_heads.track_head_plugin import Instances

import pdb
import matplotlib.pyplot as plt


class CrossLaneInteraction(nn.Module):

    def __init__(self, pc_range, inf_pc_range, embed_dims=256):
        super(CrossLaneInteraction, self).__init__()

        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.embed_dims = embed_dims

        # reference_points ---> pos_embed
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _lidar2norm(self, locs, pc_range, norm_mode='sigmoid'):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        if norm_mode not in ['sigmoid', 'inverse_sigmoid']:
            raise Exception('mode is not correct with {}'.format(norm_mode))

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        if norm_mode == 'inverse_sigmoid':
            locs = inverse_sigmoid(locs)

        return locs
    
    def _norm2lidar(self, ref_pts, pc_range, norm_mode='sigmoid'):
        """
        normalized (x,y) ---> absolute (x,y) in inf lidar coordinate system
        """
        if norm_mode not in ['sigmoid', 'inverse_sigmoid']:
            raise Exception('mode is not correct with {}'.format(norm_mode))
        if norm_mode == 'inverse_sigmoid':
            locs = ref_pts.sigmoid().clone()
        else:
            locs = ref_pts.clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs
    
    def _dis_filt(self, veh_pts, inf_pts, veh_dims):
        """
        filter according to distance
        """
        diff = torch.abs(veh_pts - inf_pts) / veh_dims
        return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1
    
    def _query_matching(self, inf_ref_pts, veh_ref_pts, veh_mask, veh_pred_dims):
        """
        inf_ref_pts: [..., 3] (xyz)
        veh_ref_pts: [..., 3] (xyz)
        veh_pred_dims: [..., 3] (dx, dy, dz)
        """
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums))
        cost_matrix.fill(1e6)

        for i in veh_mask:
            # for j in range(i,inf_nums):
            for j in range(inf_nums):
                cost_matrix[i][j] = torch.sum((veh_ref_pts[i] - inf_ref_pts[j])**2)**0.5
                if not self._dis_filt(veh_ref_pts[i], inf_ref_pts[j], veh_pred_dims[i]):
                    cost_matrix[i][j] = 1e6
        
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)

        return idx_veh, idx_inf, cost_matrix
    
    def _query_fusion(self, inf, veh, inf_idx, veh_idx, cost_matrix):
        """
        Query fusion: 
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP
        
        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """

        veh_accept_idx = []
        inf_accept_idx = []

        for i in range(len(veh_idx)):
            if cost_matrix[veh_idx[i]][inf_idx[i]] < 1e5:
                veh_accept_idx.append(veh_idx[i])
                inf_accept_idx.append(inf_idx[i])

                veh.query[veh_idx[i], self.embed_dims:] = veh.query[veh_idx[i], self.embed_dims:] + self.cross_agent_fusion(inf.query[inf_idx[i], self.embed_dims:])
        
        return veh, veh_accept_idx, inf_accept_idx
    

    def _query_complementation(self, inf, veh, inf_accept_idx):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        # supply_idx = -1
        for i in range(inf.ref_pts.shape[0]):
            if i not in inf_accept_idx:
                veh = Instances.cat([veh, inf[i]])

        return veh

    # def forward(self, inf, veh, veh2inf_rt, threshold=0.3, debug=False, name='debug-vis.jpg'):
    def forward_fusion(self, inf, veh, veh2inf_rt, threshold=0.3, debug=False, name='debug-vis.jpg'):
        """
        Query-based cross-agent interaction: only update ref_pts and query.

        inf: Instance from infrastructure
        veh: Instance from vehicle
        veh2inf_rt: calibration parameters from infrastructure to vehicle
        """
        # pdb.set_trace()

        # confidence-based query selection for inf
        # inf_mask = torch.where(inf.scores>=threshold)
        inf_mask = torch.where(inf.obj_idxes>=0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh
        inf_mask_new = torch.where(inf.obj_idxes>=0)
        
        #not care obj_idxes of inf
        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1
                
        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, self.inf_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)
            
        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(veh2inf_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # ego_selection
        remove_ego_ins = True
        if remove_ego_ins:
            H_B, H_F = -2.04, 2.04 # H = 4.084
            W_L, W_R = -0.92, 0.92 # W = 1.85
            def del_tensor_ele(arr,index):
                arr1 = arr[0:index]
                arr2 = arr[index+1:]
                return torch.cat((arr1,arr2),dim=0)

            inf_mask_new = list(inf_mask_new)
            for ii in range(len(inf_ref_pts)):
                xx, yy = inf_ref_pts[ii][0], inf_ref_pts[ii][1]
                if xx >= H_B and xx <= H_F and yy >= W_L and yy <= W_R:
                    inf_mask_new[0] = del_tensor_ele(inf_mask_new[0], ii)
                    break
            inf_mask_new = tuple(inf_mask_new)
            inf = inf[inf_mask_new]
            inf_ref_pts = inf_ref_pts[inf_mask_new]

        # matching
        veh_mask = torch.where(veh.scores >= 0.05)[0]
        veh_idx, inf_idx, cost_matrix = self._query_matching(inf_ref_pts, veh_ref_pts, veh_mask, veh.pred_boxes[..., [2,3,5]]) # veh.pred_boxes x,y,dx,dy,z,dz

        # ref_pts normalization
        inf_ref_pts = self._loc_norm(inf_ref_pts, self.pc_range)
        veh_ref_pts = self._loc_norm(veh_ref_pts, self.pc_range)
        inf.ref_pts = inf_ref_pts
        veh.ref_pts = veh_ref_pts

        # update pos_embedding according to new ref_pts
        # inf.query[..., :self.embed_dims] = self.get_pos_embedding(inf_ref_pts)
        # update ref_pts
        # inf.ref_pts = inf_ref_pts
        # cross-agent feature alignment
        inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf.query.shape[0], 1)
        inf.query[..., :self.embed_dims] = self.cross_agent_align_pos(torch.cat([inf.query[..., :self.embed_dims],inf2veh_r], -1))
        inf.query[..., self.embed_dims:] = self.cross_agent_align(torch.cat([inf.query[..., self.embed_dims:],inf2veh_r], -1))

        # cross-agent query fusion
        veh, veh_accept_idx, inf_accept_idx = self._query_fusion(inf, veh, inf_idx, veh_idx, cost_matrix)

        # cross-agent query complementation
        veh = self._query_complementation(inf, veh, inf_accept_idx)
        # pdb.set_trace()

        return veh

    def filter_inf_lanes(self, cls_score, threshold=0.05, num_things_classes=3):
        '''
        refer to _get_bboxes_single in panseg_head.py
        cls_score = inf_outputs_classes[-1, bs]
        '''
        cls_score = cls_score.sigmoid()
        indexes = list(torch.where(cls_score.view(-1) > threshold))[0]
        det_labels = indexes % num_things_classes
        bbox_index = indexes // num_things_classes
        
        return bbox_index

    def forward(self, inf_outputs_classes, inf_outputs_coords, inf_query, inf_query_pos, inf_reference,
                                    veh_outputs_classes, veh_outputs_coords, veh_query, veh_query_pos, veh_reference,
                                    veh2inf_rt, threshold=0.05):
        '''
        reference: (x, y, w, h), reference = inverse_sigmoid(reference)
        outputs_coords: (x, y, w, h), outputs_coords = outputs_coords.sigmoid()
        '''
        calib_inf2veh = np.linalg.inv(veh2inf_rt[0].cpu().numpy().T)
        calib_inf2veh = torch.tensor(calib_inf2veh).to(inf_query)

        # UniV2X TODO: hardcode for filtering inf queries with scores
        # UniV2X TODO: supposed that img num = 1
        inf_cls_scores = inf_outputs_classes[-1]
        # for img_id in range(inf_cls_scores.shape[0]):
        #     inf_bbox_index = self.filter_inf_lanes(inf_cls_scores, threshold=threshold)

        #     inf_outputs_classes = inf_outputs_classes[:, img_id, inf_bbox_index, :]
        #     inf_outputs_coords = inf_outputs_coords[:, img_id, inf_bbox_index, :]
        #     inf_query = inf_query[img_id, inf_bbox_index, :]
        #     inf_query_pos = inf_query_pos[img_id, inf_bbox_index, :]
        #     inf_reference = inf_reference[img_id, inf_bbox_index, :]
        inf_bbox_index = self.filter_inf_lanes(inf_cls_scores, threshold=threshold)

        #for trans. cost
        # print('inf lane nums: ',len(inf_bbox_index))

        inf_outputs_classes = inf_outputs_classes[:, :, inf_bbox_index, :]
        inf_outputs_coords = inf_outputs_coords[:, :, inf_bbox_index, :]
        inf_query = inf_query[:, inf_bbox_index, :]
        inf_query_pos = inf_query_pos[:, inf_bbox_index, :]
        inf_reference = inf_reference[:, inf_bbox_index, :]

        # inf_reference: inf2veh
        inf_ref_pts = torch.zeros(inf_reference.shape[0], 
                                                            inf_reference.shape[1], 
                                                            3).to(inf_query)
        inf_ref_pts[..., :2] = inf_reference[..., :2]
        for ii in range(inf_ref_pts.shape[0]):
            inf_ref_pts[ii] = self._norm2lidar(inf_ref_pts[ii], self.inf_pc_range, norm_mode='inverse_sigmoid')
            inf_ref_tmp = torch.cat((inf_ref_pts[ii], torch.ones_like(inf_ref_pts[ii][..., :1])), -1).unsqueeze(-1)
            inf_ref_pts[ii] = torch.matmul(calib_inf2veh, inf_ref_tmp).squeeze(-1)[..., :3]

            inf_ref_pts[ii] = self._lidar2norm(inf_ref_pts[ii], self.pc_range, norm_mode='inverse_sigmoid')
        inf_reference[..., :2] = inf_ref_pts[..., :2]

        # inf_bboxes: inf2veh
        inf_bboxes = torch.zeros(inf_outputs_coords.shape[0], 
                                                        inf_outputs_coords.shape[1], 
                                                        inf_outputs_coords.shape[2], 
                                                        3).to(inf_query)
        inf_bboxes[..., :2] = inf_outputs_coords[..., :2]
        for ii in range(inf_bboxes.shape[0]):
            for jj in range(inf_bboxes.shape[1]):
                inf_bboxes[ii, jj] = self._norm2lidar(inf_bboxes[ii, jj], self.inf_pc_range, norm_mode='sigmoid')
                inf_ref_tmp = torch.cat((inf_bboxes[ii, jj], torch.ones_like(inf_bboxes[ii, jj][..., :1])), -1).unsqueeze(-1)
                inf_bboxes[ii, jj] = torch.matmul(calib_inf2veh, inf_ref_tmp).squeeze(-1)[..., :3]

                inf_bboxes[ii, jj] = self._lidar2norm(inf_bboxes[ii, jj], self.pc_range, norm_mode='sigmoid')
        inf_outputs_coords[..., :2] = inf_bboxes[..., :2]

        # cross-agent feature alignment
        for ii in range(inf_query.shape[0]):
            inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf_query[ii].shape[0], 1)
            inf_query[ii] = self.cross_agent_align(torch.cat([inf_query[ii], inf2veh_r], -1))
            inf_query_pos[ii] = self.cross_agent_align_pos(torch.cat([inf_query_pos[ii], inf2veh_r], -1))

        # UniV2X TODO: directly concat inf queries and veh queries
        # UniV2X TODO: supposed that img num = 1
        inf_outputs_classes = torch.cat((veh_outputs_classes, inf_outputs_classes), dim=2)
        inf_outputs_coords = torch.cat((veh_outputs_coords, inf_outputs_coords), dim=2)
        inf_query = torch.cat((veh_query, inf_query), dim=1)
        inf_query_pos = torch.cat((veh_query_pos, inf_query_pos), dim=1)
        inf_reference = torch.cat((veh_reference, inf_reference), dim=1)

        return inf_outputs_classes, inf_outputs_coords, inf_query, inf_query_pos, inf_reference

    def get_inf2veh_query(self, inf_outputs_classes, inf_query, inf_query_pos, inf_reference,
                                                            veh2inf_rt, threshold=0.05):
        calib_inf2veh = np.linalg.inv(veh2inf_rt[0].cpu().numpy().T)
        calib_inf2veh = torch.tensor(calib_inf2veh).to(inf_query)

        # UniV2X TODO: hardcode for filtering inf queries with scores
        # UniV2X TODO: supposed that img num = 1
        inf_cls_scores = inf_outputs_classes[-1]
        inf_bbox_index = self.filter_inf_lanes(inf_cls_scores, threshold=threshold)
        inf_outputs_classes = inf_outputs_classes[:, :, inf_bbox_index, :]
        inf_query = inf_query[:, inf_bbox_index, :]
        inf_query_pos = inf_query_pos[:, inf_bbox_index, :]
        inf_reference = inf_reference[:, inf_bbox_index, :]

        # inf_reference: inf2veh
        inf_ref_pts = torch.zeros(inf_reference.shape[0],
                                                            inf_reference.shape[1],
                                                            3).to(inf_query)
        inf_ref_pts[..., :2] = inf_reference[..., :2]
        for ii in range(inf_ref_pts.shape[0]):
            inf_ref_pts[ii] = self._norm2lidar(inf_ref_pts[ii], self.inf_pc_range, norm_mode='inverse_sigmoid')
            inf_ref_tmp = torch.cat((inf_ref_pts[ii], torch.ones_like(inf_ref_pts[ii][..., :1])), -1).unsqueeze(-1)
            inf_ref_pts[ii] = torch.matmul(calib_inf2veh, inf_ref_tmp).squeeze(-1)[..., :3]

            inf_ref_pts[ii] = self._lidar2norm(inf_ref_pts[ii], self.pc_range, norm_mode='inverse_sigmoid')
        inf_reference[..., :2] = inf_ref_pts[..., :2]

        return inf_query, inf_query_pos, inf_reference
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..dense_heads.track_head_plugin import Instances

import pdb
import matplotlib.pyplot as plt


class CrossAgentSparseInteraction(nn.Module):

    def __init__(self, pc_range, inf_pc_range, embed_dims=256):
        super(CrossAgentSparseInteraction, self).__init__()

        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.embed_dims = embed_dims

        # reference_points ---> pos_embed
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims + 9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims + 9, self.embed_dims)
        # self.cross_agent_align = nn.Sequential(
        #     nn.Linear(self.embed_dims+9, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.embed_dims),
        # )
        # cross-agent feature fusion
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)
        # self.cross_agent_fusion = nn.Linear(self.embed_dims*2, self.embed_dims)
        # self.cross_agent_fusion = nn.Sequential(
        #     nn.Linear(self.embed_dims*2, self.embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims, self.embed_dims),
        # )

        # self.veh_low_score_replace = nn.Linear(2 * self.embed_dims, self.embed_dims)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _loc_norm(self, locs, pc_range):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        # from mmdet.models.utils.transformer import inverse_sigmoid

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        # locs = inverse_sigmoid(locs)

        return locs

    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        # locs = ref_pts.sigmoid().clone()
        locs = ref_pts.clone()

        locs[:, 0:1] = locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        locs[:, 1:2] = locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        locs[:, 2:3] = locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

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
        cost_matrix = np.ones((veh_nums, inf_nums)) * 1e6
        # import pdb;pdb.set_trace()
        veh_ref_pts_expanded = veh_ref_pts.unsqueeze(1).expand(-1, inf_nums, -1)
        inf_ref_pts_expanded = inf_ref_pts.unsqueeze(0).expand(veh_nums, -1, -1)
        distances = torch.sqrt(
            torch.sum((veh_ref_pts_expanded - inf_ref_pts_expanded) ** 2, dim=-1)
        )

        veh_pred_dims_expanded = (
            veh_pred_dims.unsqueeze(1).expand(-1, inf_nums, -1).exp()
        )
        diff = (
            torch.abs(veh_ref_pts_expanded - inf_ref_pts_expanded)
            / veh_pred_dims_expanded
        )
        filter_mask = (diff[..., 0] <= 1) & (diff[..., 1] <= 1) & (diff[..., 2] <= 1)

        distances = distances.detach().cpu().numpy()
        veh_mask = veh_mask.detach().cpu().numpy()
        filter_mask = filter_mask.detach().cpu().numpy()
        cost_matrix[veh_mask, :] = distances[veh_mask, :]
        cost_matrix[~filter_mask] = 1e6

        # for i in veh_mask:
        #     # for j in range(i,inf_nums):
        #     for j in range(inf_nums):
        #         cost_matrix[i][j] = torch.sum((veh_ref_pts[i] - inf_ref_pts[j])**2)**0.5
        #         if not self._dis_filt(veh_ref_pts[i], inf_ref_pts[j], veh_pred_dims[i]):
        #             cost_matrix[i][j] = 1e6

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
        mask = cost_matrix[veh_idx, inf_idx] < 1e5
        veh_accept_idx = veh_idx[mask]
        inf_accept_idx = inf_idx[mask]

        matched_veh = veh[veh_accept_idx]
        matched_inf = inf[inf_accept_idx]
        matched_veh.query_feats = matched_veh.query_feats + self.cross_agent_fusion(
            matched_inf.query_feats
        )

        # for i in range(len(veh_idx)):
        #     if cost_matrix[veh_idx[i]][inf_idx[i]] < 1e5:
        #         veh_accept_idx.append(veh_idx[i])
        #         inf_accept_idx.append(inf_idx[i])

        #         # if veh.scores[veh_idx[i]] < inf.scores[inf_idx[i]]:
        #         #     veh.scores[veh_idx[i]] = inf.scores[inf_idx[i]]
        #         #     veh.ref_pts[veh_idx[i]] = inf.ref_pts[inf_idx[i]]
        #         #     veh.query[veh_idx[i], :self.embed_dims] = inf.query[inf_idx[i], :self.embed_dims]

        #         # veh.query[veh_idx[i], self.embed_dims:] = self.cross_agent_fusion(torch.cat([veh.query[veh_idx[i], self.embed_dims:], inf.query[inf_idx[i], self.embed_dims:]], -1))
        #         veh.query[veh_idx[i], self.embed_dims :] = veh.query[
        #             veh_idx[i], self.embed_dims :
        #         ] + self.cross_agent_fusion(inf.query[inf_idx[i], self.embed_dims :])

        # return veh, veh_accept_idx, inf_accept_idx
        return matched_veh, veh_accept_idx, inf_accept_idx

    def _query_complementation(self, inf, veh, inf_accept_idx, veh_accept_idx, fused):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        veh_num = len(veh)
        inf_num = len(inf)

        mask = torch.ones(veh_num, dtype=bool)
        mask[veh_accept_idx] = False
        unmatched_veh = veh[mask]

        mask = torch.ones(inf_num, dtype=bool)
        mask[inf_accept_idx] = False
        unmatched_inf = inf[mask]
        # print('unmatched_inf obj:', unmatched_inf.obj_idxes)
        # import pdb;pdb.set_trace()
        res_instances = Instances((1, 1))
        res_instances = Instances.cat([res_instances, fused])
        res_instances = Instances.cat([res_instances, unmatched_inf])

        select_num = veh_num - inf_num
        # print('----------')
        # print(len(fused), inf_num)
        _, topk_indexes = torch.topk(unmatched_veh.scores, select_num, dim=0)
        res_instances = Instances.cat([res_instances, unmatched_veh[topk_indexes]])

        return res_instances

    def _vis(
        self, inf_pts, veh_pts, veh_score, vis_threshold=0.3, name='debug-vis.jpg'
    ):
        """
        Visualization for debug

        inf_pts: reference points after coordinate transformation for inf
        veh_pts: reference points for veh
        """

        temp = inf_pts.contiguous().cpu().detach().numpy()
        plt.scatter(temp[:, 0], temp[:, 1], c='r')

        veh_mask = torch.where(veh_score >= vis_threshold)
        veh_pts = veh_pts[veh_mask]
        temp = veh_pts.contiguous().cpu().detach().numpy()
        plt.scatter(temp[:, 0], temp[:, 1], c='g')

        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        plt.savefig(f"test/debug/{name}")

        plt.close()

    def forward(
        self, inf, veh, veh2inf_rt, threshold=0.3, debug=False, name='debug-vis.jpg'
    ):
        """
        Query-based cross-agent interaction: only update ref_pts and query.

        inf: Instance from infrastructure
        veh: Instance from vehicle
        veh2inf_rt: calibration parameters from infrastructure to vehicle
        """
        # import pdb; pdb.set_trace()

        # confidence-based query selection for inf
        # inf_mask = torch.where(inf.scores>=threshold)
        inf_mask = torch.where(inf.obj_idxes >= 0)
        inf = inf[inf_mask]
        if len(inf) == 0:
            return veh, 0
        inf_mask_new = torch.where(inf.obj_idxes >= 0)

        # not care obj_idxes of inf
        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1

        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, self.inf_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)
        if debug:
            self._vis(
                inf_ref_pts,
                veh_ref_pts,
                veh.scores,
                vis_threshold=1.0,
                name="inf-" + name,
            )

        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat(
            (inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1
        ).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # # For griffin: refine ref_pts to the mean height of vehicle
        # drone_pos = calib_inf2veh[:3, 3]
        # refine_ratio = (drone_pos[2] - veh_ref_pts[:, 2].mean()) / (
        #     drone_pos[2] - inf_ref_pts[:, 2]
        # )
        # inf_ref_pts = (inf_ref_pts - drone_pos) * refine_ratio.unsqueeze(1) + drone_pos

        # ego_selection
        remove_ego_ins = False
        if remove_ego_ins:
            H_B, H_F = -2.04, 2.04  # H = 4.084
            W_L, W_R = -0.92, 0.92  # W = 1.85

            def del_tensor_ele(arr, index):
                arr1 = arr[0:index]
                arr2 = arr[index + 1 :]
                return torch.cat((arr1, arr2), dim=0)

            inf_mask_new = list(inf_mask_new)
            for ii in range(len(inf_ref_pts)):
                xx, yy = inf_ref_pts[ii][0], inf_ref_pts[ii][1]
                if xx >= H_B and xx <= H_F and yy >= W_L and yy <= W_R:
                    inf_mask_new[0] = del_tensor_ele(inf_mask_new[0], ii)
                    break
            inf_mask_new = tuple(inf_mask_new)
            inf = inf[inf_mask_new]
            inf_ref_pts = inf_ref_pts[inf_mask_new]

        if debug:
            self._vis(
                inf_ref_pts, veh_ref_pts, veh.scores, vis_threshold=0.0, name=name
            )

        # matching
        veh_mask = torch.where(veh.scores >= 0.05)[0]
        veh_idx, inf_idx, cost_matrix = self._query_matching(
            inf_ref_pts, veh_ref_pts, veh_mask, veh.pred_boxes[..., [2, 3, 5]]
        )  # veh.pred_boxes x,y,dx,dy,z,dz

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
        inf2veh_r = (
            calib_inf2veh[:3, :3].reshape(1, 9).repeat(inf.query_feats.shape[0], 1)
        )
        inf.query_embeds = self.cross_agent_align_pos(
            torch.cat([inf.query_embeds, inf2veh_r], -1)
        )
        inf.query_feats = self.cross_agent_align(
            torch.cat([inf.query_feats, inf2veh_r], -1)
        )

        # cross-agent query fusion
        fused, veh_accept_idx, inf_accept_idx = self._query_fusion(
            inf, veh, inf_idx, veh_idx, cost_matrix
        )

        # cross-agent query complementation
        veh = self._query_complementation(
            inf, veh, inf_accept_idx, veh_accept_idx, fused
        )
        # pdb.set_trace()

        return veh, len(inf)

    def forward_only_inf(
        self, inf, veh, veh2inf_rt, threshold=0.3, debug=False, name='debug-vis.jpg'
    ):
        """
        Query-based cross-agent interaction: only update ref_pts and query.

        inf: Instance from infrastructure
        veh: Instance from vehicle
        veh2inf_rt: calibration parameters from infrastructure to vehicle
        """
        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, self.inf_pc_range)

        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(veh2inf_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat(
            (inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1
        ).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # ref_pts normalization
        inf_ref_pts = self._loc_norm(inf_ref_pts, self.inf_pc_range)
        # veh_ref_pts = self._loc_norm(veh_ref_pts)
        inf.ref_pts = inf_ref_pts
        # veh.ref_pts = veh_ref_pts

        # update pos_embedding according to new ref_pts
        inf.query[..., : self.embed_dims] = self.get_pos_embedding(inf_ref_pts)
        # update ref_pts
        inf.ref_pts = inf_ref_pts
        # cross-agent feature alignment
        inf2veh_r = calib_inf2veh[:3, :3].reshape(1, 9).repeat(inf.query.shape[0], 1)
        inf.query[..., self.embed_dims :] = self.cross_agent_align(
            torch.cat([inf.query[..., self.embed_dims :], inf2veh_r], -1)
        )

        return inf

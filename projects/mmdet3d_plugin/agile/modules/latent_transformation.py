import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Linear

class LatentTransformation(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 head=16,
                 rot_dims=6,
                 trans_dims=3,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 inf_pc_range=[0, -51.2, -5.0, 102.4, 51.2, 3.0],
                 ):
        super(LatentTransformation, self).__init__()
        self.embed_dims = embed_dims
        self.head = head
        self.rot_dims = rot_dims
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range

        rot_final_dim = int((embed_dims / head) * (embed_dims / head) * head)
        trans_final_dim = embed_dims

        layers = []
        dims = [rot_dims, embed_dims, embed_dims, rot_final_dim]
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.rot_mlp = nn.Sequential(*layers)

        layers = []
        dims = [trans_dims, embed_dims, embed_dims, trans_final_dim]
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.trans_mlp = nn.Sequential(*layers)

        self.feat_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.embed_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.motion_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )

        self.feat_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.embed_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.motion_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
    
    def continuous_rot(self, rot):
        ret = rot[:, :2].clone()
        ret = ret.reshape(1, -1)
        return ret

    def fill_tensor(self, original_tensor):
        h = self.head
        k = self.embed_dims // self.head
        d = self.embed_dims
        
        # 生成块的基础索引 (每个块的左上角行/列坐标)
        blocks_indices = torch.arange(h, device=original_tensor.device) * k
        
        # 生成块内的行和列偏移 (0 到 k-1)
        offset = torch.arange(k, device=original_tensor.device)
        rows_offset, cols_offset = torch.meshgrid(offset, offset)
        
        # 计算所有块的全局坐标（利用广播机制）
        base_rows = blocks_indices.view(h, 1, 1)          # Shape: (h, 1, 1)
        global_rows = base_rows + rows_offset.unsqueeze(0) # Shape: (h, k, k)
        base_cols = blocks_indices.view(h, 1, 1)
        global_cols = base_cols + cols_offset.unsqueeze(0) # Shape: (h, k, k)
        
        # 展平坐标和原始数据
        all_rows = global_rows.reshape(-1)                # Shape: (h*k*k, )
        all_cols = global_cols.reshape(-1)                # Shape: (h*k*k, )
        data = original_tensor.view(-1)                   # Shape: (h*k*k, )
        
        # 创建目标张量并一次性赋值
        target_tensor = torch.zeros((d, d), device=original_tensor.device)
        target_tensor[all_rows, all_cols] = data
        
        return target_tensor

    def transform_pts(self, points, transformation):
        # relative -> absolute (in inf pc range)
        locs = points.clone()
        locs[:, 0:1] = (locs[:, 0:1] * (self.inf_pc_range[3] - self.inf_pc_range[0]) + self.inf_pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (self.inf_pc_range[4] - self.inf_pc_range[1]) + self.inf_pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (self.inf_pc_range[5] - self.inf_pc_range[2]) + self.inf_pc_range[2])

        # transformation
        locs = torch.cat((locs, torch.ones_like(locs[..., :1])), -1).unsqueeze(-1)
        locs = torch.matmul(transformation, locs).squeeze(-1)[..., :3]
        
        # filter
        mask = (self.pc_range[0] <= locs[:, 0]) & (locs[:, 0] <= self.pc_range[3]) & \
                    (self.pc_range[1] <= locs[:, 1]) & (locs[:, 1] <= self.pc_range[4])
        locs = locs[mask]
        # absolute -> relative (in veh pc range)
        locs[..., 0:1] = (locs[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        return locs, mask
    
    def forward(self, instances, veh2inf_rt):
        # import pdb;pdb.set_trace()
        calib_inf2veh = np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        calib_inf2veh = instances['ref_pts'].new_tensor(calib_inf2veh)
        rot = calib_inf2veh[:3, :3].clone()
        trans = calib_inf2veh[:3, 3:4].clone()

        if self.rot_dims == 6:
            con_rot = self.continuous_rot(rot)
            assert con_rot.size(1) == 6
        trans = trans.reshape(1, -1)

        rot_para = self.rot_mlp(con_rot)
        trans_para = self.trans_mlp(trans) #(1, d)
        rot_mat = self.fill_tensor(rot_para) # (d, d)
        # import pdb;pdb.set_trace()
        instances['ref_pts'], mask = self.transform_pts(instances['ref_pts'], calib_inf2veh)
        instances['query_feats'] = instances['query_feats'][mask]
        instances['query_embeds'] = instances['query_embeds'][mask]
        instances['cache_motion_feats'] = instances['cache_motion_feats'][mask]

        identity_query_feats = instances['query_feats'].clone()
        identity_query_embeds = instances['query_embeds'].clone()
        identity_cache_motion_feats = instances['cache_motion_feats'].clone()

        instances['query_feats'] = self.feat_output_proj((self.feat_input_proj(instances['query_feats']) @ rot_mat.T + trans_para) + identity_query_feats)
        instances['query_embeds'] = self.embed_output_proj((self.embed_input_proj(instances['query_embeds']) @ rot_mat.T + trans_para) + identity_query_embeds)
        instances['cache_motion_feats'] = self.motion_output_proj((self.motion_input_proj(instances['cache_motion_feats']) @ rot_mat.T + trans_para) + identity_cache_motion_feats)
        
        return instances

        
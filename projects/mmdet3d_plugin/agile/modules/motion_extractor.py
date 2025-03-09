import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

class MotionExtractor(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 mlp_channels=(3, 64, 64, 256),
                 ):
        super(MotionExtractor, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_channels = mlp_channels

        pointnet = []
        for i in range(len(self.mlp_channels) - 1):
            pointnet.append(Linear(self.mlp_channels[i], self.mlp_channels[i+1]))
            pointnet.append(nn.LayerNorm(self.mlp_channels[i+1]))
            pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*pointnet)

        velonet = []
        for i in range(len(self.mlp_channels) - 1):
            velonet.append(Linear(self.mlp_channels[i], self.mlp_channels[i+1]))
            velonet.append(nn.LayerNorm(self.mlp_channels[i+1]))
            velonet.append(nn.ReLU())
        self.velonet = nn.Sequential(*velonet)

        fusion_net = []
        fusion_net.append(Linear(self.embed_dims * 2, self.embed_dims))
        fusion_net.append(nn.LayerNorm(self.embed_dims))
        fusion_net.append(nn.ReLU())
        self.fusion_net = nn.Sequential(*fusion_net)

    def forward(self, track_instances, img_metas):
        # import pdb;pdb.set_trace()
        pred_bboxes = track_instances.cache_bboxes.clone()
        # cx, cy, cz, w, l, h, rot, vx, vy
        decode_bboxes = denormalize_bbox(pred_bboxes, None)
        bboxes_geometric = decode_bboxes[:, 0:7]
        bbox = LiDARInstance3DBoxes(bboxes_geometric,
                                    box_dim=7)
        corners = bbox.corners
        center = bbox.gravity_center
        normalized_corners = corners - center.unsqueeze(1)
        point_feat = self.pointnet(normalized_corners)
        point_feat = torch.max(point_feat, dim=1)[0]

        bboxes_motion = decode_bboxes[:, 6:9]
        motion_feat = self.velonet(bboxes_motion)

        fusion_feat = torch.cat([point_feat, motion_feat], dim=1)
        fusion_feat = self.fusion_net(fusion_feat)

        track_instances.cache_motion_feats = fusion_feat.clone()
        return track_instances


# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN -> R50
# smaller BEV: 200*200 -> 50*50
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
post_center_range = [-61.2, -61.2, -10, 61.2, 61.2, 10]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

class_names = [
    "car",
    "bicycle",
    "pedestrian",
]

# class_range for eva
class_range = {
    "car": 50,
    "bicycle": 50,
    "pedestrian": 50,
}

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

dataset_prefix = "griffin_50scenes_25m"  # * different for every dataset
v2x_side = "early-fusion"
num_cams = 9

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
num_query = 900

# Other settings
train_gt_iou_threshold = 0.3

# stream training
num_each_seq = 5  # 0: keep original sequences
seq_mode = True
queue_length = 1  # when stream training, queue_length=1

num_gpus = 4
batch_size = 2  # batch on each gpu
train_sample_num = 5510  # * different for every dataset
num_iters_per_epoch = train_sample_num // (num_gpus * batch_size)
num_epochs = 48

model = dict(
    type="Agile",
    seq_mode=seq_mode,
    batch_size=batch_size,
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=num_query,
    num_classes=len(class_names),
    pc_range=point_cloud_range,
    post_center_range=post_center_range,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    freeze_img_backbone=True,
    freeze_img_neck=False,
    freeze_bn=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    # score_thresh=0.3,
    # filter_score_thresh=0.25,
    # miss_tolerance=21,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),  # hyper-param for query dropping mentioned in MOTR
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=len(class_names),
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_past_traj_weight=0.0,
    ),  # loss cfg for tracking
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=num_query,
        num_classes=len(class_names),
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type="PerceptionTransformer",
            num_cams=num_cams,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            num_cams=num_cams,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=post_center_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names),
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)

file_client_args = dict(backend="disk")

dataset_type = "GriffinDataset"
data_root = f"./datasets/{dataset_prefix}/griffin-nuscenes/{v2x_side}/"
info_root = f"./data/infos/{dataset_prefix}/{v2x_side}/"
ann_file_train = info_root + "griffin_infos_train.pkl"
ann_file_val = info_root + "griffin_infos_val.pkl"
split_datas_file = f"./data/split_datas/{dataset_prefix}.json"

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFilesInCeph",
        to_float32=True,
        file_client_args=file_client_args,
        img_root=data_root,
    ),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=False,  # occ_flow gt
        with_ins_inds_3d=True,  # ins_inds
        ins_inds_add_1=True,  # ins_inds start from 1
    ),
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
        ],
    ),
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesInCeph',
        to_float32=True,
        file_client_args=file_client_args,
        img_root=data_root,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type='LoadAnnotations3D_E2E',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=False,
        with_ins_inds_3d=True,
        ins_inds_add_1=True,  # ins_inds start from 1
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D",
                keys=[
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "gt_inds",
                    "img",
                    "timestamp",
                    "l2g_r_mat",
                    "l2g_t",
                ],
            ),
        ],
    ),
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,  # 0 is single subprocess
    train=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        num_each_seq=num_each_seq,
        seq_mode=seq_mode,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        split_datas_file=split_datas_file,
        v2x_side=v2x_side,
        class_range=class_range,
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track'],
        split_datas_file=split_datas_file,
        v2x_side=v2x_side,
        class_range=class_range,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_val,  # Todo: temp: use val for test
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'track'],
        split_datas_file=split_datas_file,
        v2x_side=v2x_side,
        class_range=class_range,
    ),
    shuffler_sampler=dict(type="InfiniteGroupEachSampleInBatchSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# 8	    2e-4
# 16	4e-4
# 32	6e-4
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

workflow = [('train', 1)]
eval_epoch_interval = 3
evaluation = dict(
    interval=num_iters_per_epoch * eval_epoch_interval, pipeline=test_pipeline
)
runner = dict(type="IterBasedRunner", max_iters=num_epochs * num_iters_per_epoch)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='v2xtrack',
                name=f'{dataset_prefix}_{v2x_side}_tiny_r50_bs{batch_size}x{num_gpus}_{num_epochs}e_3cls',
            ),
        ),
    ],
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
checkpoint_config = dict(
    interval=num_iters_per_epoch * eval_epoch_interval, max_keep_ckpts=3
)

work_dir = None
load_from = "ckpts/bevformer_tiny_epoch_24.pth"
resume_from = None

find_unused_parameters = True

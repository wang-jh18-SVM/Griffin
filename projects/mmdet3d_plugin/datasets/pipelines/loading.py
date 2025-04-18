import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet3d.core.points import BasePoints, get_points_type
import os


@PIPELINES.register_module()
class LoadPointsFromFile_E2E(LoadPointsFromFile):
    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=...,
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend='disk'),
        pts_root='',
    ):

        super().__init__(
            coord_type, load_dim, use_dim, shift_height, use_color, file_client_args
        )

        self.pts_root = pts_root

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = os.path.join(self.pts_root, results['pts_filename'])
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results['points'] = points

        return results


@PIPELINES.register_module()
class LoadInfInformation(object):
    """Load inf query from a files.

    Expects results['sample_idx_inf'] to be a str.

    Args:
        load_file_root (str): the root path
    """

    def __init__(
        self, load_file_root=False, file_client_args=dict(backend='disk'), keys=None
    ):
        self.load_file_root = load_file_root
        self.keys = keys
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)

    def __call__(self, results):
        """Call function to load inf information.

        Args:
            results (dict): Result dict containing inf information.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - inf_infor (dict) : mulitple inf information
        """
        inf_idx = (
            results['sample_idx_inf']
            if 'sample_idx_inf' in results
            else results['sample_idx']
        )
        filename = os.path.join(self.load_file_root, inf_idx + '.pkl')
        infos = mmcv.load(filename)

        for k in self.keys:
            results[k] = infos.get(k)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromFilesInCeph(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        to_float32=False,
        color_type='unchanged',
        file_client_args=dict(backend='disk'),
        img_root='',
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.img_root = img_root

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (list of str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        images_multiView = []
        filename = results['img_filename']
        for img_path in filename:
            img_path = os.path.join(self.img_root, img_path)
            if self.file_client_args['backend'] == 'petrel':
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
            elif self.file_client_args['backend'] == 'disk':
                img = mmcv.imread(img_path, self.color_type)

            # For Griffin
            if img.shape[2] == 4:
                img = img[:, :, :3]

            images_multiView.append(img)
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            # [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            images_multiView,
            axis=-1,
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D_E2E(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(
        self,
        with_future_anns=False,
        with_ins_inds_3d=False,
        ins_inds_add_1=False,  # NOTE: make ins_inds start from 1, not 0
        with_forecasting=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.with_future_anns = with_future_anns
        self.with_ins_inds_3d = with_ins_inds_3d
        self.with_forecasting = with_forecasting

        self.ins_inds_add_1 = ins_inds_add_1

    def _load_future_anns(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        # gt_valid_flags = []
        gt_vis_tokens = []

        for ann_info in results['occ_future_ann_infos']:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info['gt_bboxes_3d'])
                gt_labels_3d.append(ann_info['gt_labels_3d'])

                ann_gt_inds = ann_info['gt_inds']
                if self.ins_inds_add_1:
                    ann_gt_inds += 1
                    # NOTE: sdc query is changed from -10 -> -9
                gt_inds_3d.append(ann_gt_inds)

                # gt_valid_flags.append(ann_info['gt_valid_flag'])
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                # invalid frame
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                # gt_valid_flags.append(None)
                gt_vis_tokens.append(None)

        results['future_gt_bboxes_3d'] = gt_bboxes_3d
        # results['future_bbox3d_fields'].append('gt_bboxes_3d')  # Field is used for augmentations, not needed here
        results['future_gt_labels_3d'] = gt_labels_3d
        results['future_gt_inds'] = gt_inds_3d
        # results['future_gt_valid_flag'] = gt_valid_flags
        results['future_gt_vis_tokens'] = gt_vis_tokens

        return results

    def _load_ins_inds_3d(self, results):
        ann_gt_inds = results['ann_info']['gt_inds'].copy()  # TODO: note here

        # NOTE: Avoid gt_inds generated twice
        results['ann_info'].pop('gt_inds')

        if self.ins_inds_add_1:
            ann_gt_inds += 1
        results['gt_inds'] = ann_gt_inds
        return results

    def _load_forecasting(self, results):
        """Private function to load forecasting annotations"""
        results['gt_forecasting_locs'] = results['ann_info']['gt_forecasting_locs']
        results['gt_forecasting_masks'] = results['ann_info']['gt_forecasting_masks']
        results['gt_forecasting_types'] = results['ann_info']['gt_forecasting_types']
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_future_anns:  # False
            results = self._load_future_anns(results)
        if self.with_ins_inds_3d:  # True
            results = self._load_ins_inds_3d(results)
        if self.with_forecasting:
            results = self._load_forecasting(results)
        # Generate ann for plan
        if 'occ_future_ann_infos_for_plan' in results.keys():
            results = self._load_future_anns_plan(results)

        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str += f'{indent_str}with_future_anns={self.with_future_anns}, '
        repr_str += f'{indent_str}with_ins_inds_3d={self.with_ins_inds_3d}, '

        return repr_str

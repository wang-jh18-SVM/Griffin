import argparse
import torch
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.agile.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np

warnings.filterwarnings("ignore")

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
    'traffic_cone': 'traffic_cone',
}

CLASSES = (
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default='output/results.pkl',
        help='output result file in pickle format',
    )
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed',
    )
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server',
    )
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.',
    )
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.',
    )
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--input-path', type=str, default='')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options'
        )
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def convert_spd2univ2x(input_path, dst_file_path, val_pkl_file):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type

    pkl_frame_id_list = []
    pkl_file_data = mmcv.load(val_pkl_file)
    for cur_data in pkl_file_data['infos']:
        if "griffin" in val_pkl_file:
            pkl_frame_id_list.append(
                str(int(int(cur_data['token'].split('_')[-1]) / 1e5))
            )
        else:
            pkl_frame_id_list.append(cur_data['token'])

    box_type_3d, box_mode_3d = get_box_type('LiDAR')
    results = []

    total_spd_results = []
    file_list = os.listdir(input_path)
    for file_name in file_list:
        file_path = os.path.join(input_path, file_name)
        with open(file_path, 'r') as f:
            cur_file_lines = f.readlines()

        # cur_file_content = []
        for line in cur_file_lines:
            line = line.strip().split()
            # cur_file_content.append(line)
            total_spd_results.append(line)
    total_spd_results = np.array(total_spd_results)

    for frame_id in pkl_frame_id_list:
        cur_frame_datas = total_spd_results[total_spd_results[:, 0] == frame_id]

        if len(cur_frame_datas) <= 0:
            print('cur frame no results: ', frame_id)
            detection = {'sample_id': frame_id}
            results.append(detection)
            continue

        frame_id = frame_id
        boxes_3d = []
        scores_3d = []
        labels_3d = []
        track_ids = []

        locs = []  # xyz
        dims = []  # wlh
        rots = []
        velocity = []
        others = []
        for cur_data in cur_frame_datas:
            locs.append([cur_data[17], cur_data[18], cur_data[19]])
            dims.append([cur_data[11], cur_data[12], cur_data[10]])
            if "griffin" in val_pkl_file:
                rot = np.float32(cur_data[20])
            else:
                rot = -np.float32(cur_data[20]) - np.pi / 2

            rots.append([rot])

            velocity.append([0, 0])
            others.append([0])

            scores_3d.append(cur_data[23])
            labels_3d.append(cur_data[1])
            track_ids.append(cur_data[2])

        boxes_3d = np.array(
            np.concatenate([locs, dims, rots, velocity, others], axis=1),
            dtype=np.float32,
        )

        boxes_3d = LiDARInstance3DBoxes(
            boxes_3d, box_dim=boxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(box_mode_3d)

        new_labels_3d = []
        for cat in labels_3d:
            new_cat = class_names_nuscenes_mappings[cat]
            if new_cat in CLASSES:
                new_labels_3d.append(CLASSES.index(new_cat))
            else:
                new_labels_3d.append(-1)

        scores_3d = np.array(scores_3d, dtype=np.float32)
        new_labels_3d = np.array(new_labels_3d, dtype=np.float32)
        track_ids = np.array(track_ids, dtype=np.float32)

        scores_3d = torch.tensor(scores_3d, dtype=torch.float32)
        new_labels_3d = torch.tensor(new_labels_3d, dtype=torch.float32)
        track_ids = torch.tensor(track_ids, dtype=torch.float32)

        detection = {}
        detection['boxes_3d'] = boxes_3d
        detection['scores_3d'] = scores_3d
        detection['labels_3d'] = new_labels_3d
        detection['track_ids'] = track_ids
        detection['sample_id'] = frame_id

        results.append(detection)

    # with open(dst_file_path,'w') as f:
    mmcv.dump(results, dst_file_path)

    return results


def main():
    args = parse_args()

    args.gpus = 1
    args.gpu_ids = [0]
    args.eval = "bbox"
    args.show = False
    args.eval = True
    args.inference = False
    args.is_spd_results = True

    print('args.config: ', args.config)
    print('args.input_path: ', args.input_path)
    print('args.out: ', args.out)
    print('args.launcher: ', args.launcher)

    current_folder_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(f'{current_folder_path}/../..')
    curDirectory = os.getcwd()
    print(curDirectory)

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        'Please specify at least one operation (save/eval/format/show the '
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg['custom_imports'])

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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # print('**cfg.dist_params: ', cfg.dist_params)
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # if 'save_track_query_file' in cfg.data.test and 'save_track_query' in cfg.data.test:
    #     save_track_query_file = cfg.data.test['save_track_query_file']
    #     if cfg.data.test['save_track_query'] and osp.exists(save_track_query_file):
    #         os.system(f'rm -rf {save_track_query_file}')

    if args.inference:
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE

    if not distributed:
        if args.inference:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            if not args.is_spd_results:
                outputs = mmcv.load(args.out)
            else:
                # convert spd to univ2x for eval
                outputs = convert_spd2univ2x(
                    args.input_path, args.out, cfg.data.val.ann_file
                )
    else:
        if args.inference:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs = custom_multi_gpu_test(
                model, data_loader, args.tmpdir, args.gpu_collect
            )
        else:
            raise NotImplementedError(
                "Please set args.inference to run distributed evaluation"
            )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            # assert False
            mmcv.dump(outputs, args.out)
            # outputs = mmcv.load(args.out)

            # save track query
            if hasattr(cfg, 'save_track_query'):
                if cfg.save_track_query:
                    if hasattr(cfg, 'save_track_query_file_test'):
                        save_path = cfg.save_track_query_file_test

                        inf_track_query_info = {}
                        for result in outputs['bbox_results']:
                            inf_track_query_info.update(result['inf_track_qry'])

                        mmcv.dump(inf_track_query_info, save_path)
            # ##save occ
            # if hasattr(cfg,'is_save_occ_prob'):
            #     if cfg.is_save_occ_prob:
            #         if hasattr(cfg,'save_occ_prob_file_test'):
            #             save_path = cfg.save_occ_prob_file_test
            #             print('ywx test: len(occ_prob): ',len(occ_prob))
            #             mmcv.dump(occ_prob,save_path)

        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join(
            'test',
            args.config.split('/')[-1].split('.')[-2],
            time.ctime().replace(' ', '_').replace(':', '_'),
        )
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
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

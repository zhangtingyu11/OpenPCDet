import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from eval_utils import eval_utils

# VoxelRCNN baseline (epoch 80, with standard augmentation)
# These are the actual fully-trained VoxelRCNN results, not frozen eval
VOXELRCNN_BASELINE = {
    '3d_AP_R40_easy': 92.42,
    '3d_AP_R40_moderate': 85.03,
    '3d_AP_R40_hard': 82.70,
    '3d_AP_R11_easy': 89.51,
    '3d_AP_R11_moderate': 84.02,
    '3d_AP_R11_hard': 78.81,
}

# Key patterns in the result_dict returned by KITTI eval
# R40 keys: Car_3d/easy_R40, Car_3d/moderate_R40, Car_3d/hard_R40
# R11 keys: commented out in eval.py, so we extract from the printed result_str


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False)
    parser.add_argument('--epochs', type=int, default=None, required=False)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--extra_tag', type=str, default='default')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    parser.add_argument('--fix_random_seed', type=int, default=666)
    parser.add_argument('--ckpt_save_interval', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_ckpt_save_num', type=int, default=10)
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--max_waiting_mins', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs_to_eval', type=int, default=10)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False)
    parser.add_argument('--logger_iter_interval', type=int, default=50)
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300)
    parser.add_argument('--wo_gpu_stat', action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
    else:
        _, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    common_utils.set_random_seed(args.fix_random_seed + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=args.fix_random_seed
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()
    logger.info('----------- Model created, param count: %d -----------' % sum([m.numel() for m in model.parameters()]))
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # Build test dataloader for per-epoch evaluation
    logger.info("----------- Create test dataloader for per-epoch eval -----------")
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('VoxelRCNN baseline metrics for comparison (fully trained, epoch 80):')
    logger.info('  3D AP_R40: Easy=%.2f  Moderate=%.2f  Hard=%.2f' % (
        VOXELRCNN_BASELINE['3d_AP_R40_easy'],
        VOXELRCNN_BASELINE['3d_AP_R40_moderate'],
        VOXELRCNN_BASELINE['3d_AP_R40_hard'],
    ))
    logger.info('  3D AP_R11: Easy=%.2f  Moderate=%.2f  Hard=%.2f' % (
        VOXELRCNN_BASELINE['3d_AP_R11_easy'],
        VOXELRCNN_BASELINE['3d_AP_R11_moderate'],
        VOXELRCNN_BASELINE['3d_AP_R11_hard'],
    ))

    def post_epoch_eval(model, trained_epoch):
        """Evaluate model after each epoch and compare with VoxelRCNN baseline."""
        logger.info('=' * 70)
        logger.info('Per-epoch evaluation for epoch %d' % trained_epoch)
        logger.info('=' * 70)

        cur_result_dir = eval_output_dir / ('epoch_%s' % trained_epoch) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        result_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, trained_epoch, logger,
            dist_test=dist_train, result_dir=cur_result_dir
        )

        # Log detailed comparison with VoxelRCNN baseline
        logger.info('-' * 70)
        logger.info('Epoch %d vs VoxelRCNN Baseline (epoch 80) - 3D Detection AP (Car):' % trained_epoch)
        logger.info('%-20s %12s %12s %12s' % ('Metric', 'CLOCs', 'VoxelRCNN', 'Diff'))
        logger.info('-' * 70)

        # R40 comparisons (these keys exist in result_dict)
        for diff_name, diff_label in [('easy', 'Easy'), ('moderate', 'Moderate'), ('hard', 'Hard')]:
            r40_key = 'Car_3d/%s_R40' % diff_name
            second_key = '3d_AP_R40_%s' % diff_name
            if r40_key in result_dict:
                clocs_val = result_dict[r40_key]
                second_val = VOXELRCNN_BASELINE[second_key]
                diff = clocs_val - second_val
                logger.info('%-20s %12.2f %12.2f %+12.2f' % (
                    '3D AP_R40 (%s)' % diff_label, clocs_val, second_val, diff))

        # Also log other available metrics for reference
        logger.info('-' * 70)
        logger.info('All 3D metrics from result_dict:')
        for k, v in sorted(result_dict.items()):
            if '3d' in k.lower():
                logger.info('  %s: %.4f' % (k, v))

        if tb_log is not None:
            for k, v in result_dict.items():
                if isinstance(v, (int, float)):
                    tb_log.add_scalar('eval/' + k, v, trained_epoch)

        model.train()
        logger.info('=' * 70)

    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        post_epoch_callback=post_epoch_eval,
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()

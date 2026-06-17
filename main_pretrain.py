# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time

import mae_st.util.env
import mae_st.util.misc as misc
import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from mae_st import models_mae
from mae_st.engine_pretrain import train_one_epoch
from mae_st.util.kinetics import Kinetics
from mae_st.util.prostate import (
    PROSTATE_CACHE_SUFFIXES,
    PROSTATE_COHORTS,
    ProstatePatchDataset,
    infer_chunk_size,
    resolve_loco_cohorts,
)
from mae_st.util.misc import NativeScalerWithGradNormCount as NativeScaler
from mae_st.util.pretrained_2d import (
    DEFAULT_2D_CKPT_DIR,
    load_2d_pretrained_weights,
)
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--path_to_data_dir",
        default="",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="How often to print training throughput and bottleneck diagnostics.",
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")

    # Video related configs
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument(
        "--decoder_arch",
        default="custom",
        choices=["custom", "vit-small"],
        help=(
            "Decoder preset. Use 'vit-small' for embed_dim=384, "
            "depth=12, num_heads=6. Explicit decoder_* args are ignored "
            "when a preset is selected."
        ),
    )
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=4, type=int)
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
    )
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--trunc_init",
        action="store_true",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.set_defaults(fp32=True)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow NVIDIA TF32 matmul/cuDNN kernels while keeping fp32 training.",
    )
    parser.add_argument("--no_tf32", action="store_false", dest="allow_tf32")
    parser.set_defaults(allow_tf32=True)
    parser.add_argument(
        "--jitter_scales_relative",
        default=[0.5, 1.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--jitter_aspect_relative",
        default=[0.75, 1.3333],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--pred_t_dim",
        type=int,
        default=8,
    )
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)

    # Prostate 3D patch data. Kinetics remains the default to preserve
    # compatibility with the original MAE-ST training command.
    parser.add_argument(
        "--data_source",
        default="kinetics",
        choices=["kinetics", "prostate"],
        help="Training data source.",
    )
    parser.add_argument(
        "--metadata_cache_dir",
        default="/newdisk/individuals/renao/3DFM/Prostate/metadata_caches",
        help="Directory containing cache_{cohort}_{suffix}.pkl files.",
    )
    parser.add_argument(
        "--prostate_cohorts",
        default=list(PROSTATE_COHORTS),
        nargs="+",
        help="Available prostate cohorts.",
    )
    parser.add_argument(
        "--leave_one_cohort",
        default="",
        choices=[""] + list(PROSTATE_COHORTS),
        help="Held-out cohort for leave-one-cohort-out pretraining.",
    )
    parser.add_argument(
        "--prostate_patch_config",
        default="p128_s0",
        choices=list(PROSTATE_CACHE_SUFFIXES),
        help=(
            "Metadata cache suffix: p128_s0 for non-overlap 128 chunks, "
            "p256_s0 for non-overlap 256 chunks sampled from overlap source, "
            "or p256_s128 for overlap 256 chunks."
        ),
    )
    parser.add_argument(
        "--prostate_no_normalize",
        action="store_true",
        help="Disable 0.45/0.225 RGB normalization for prostate patches.",
    )

    parser.add_argument(
        "--use_2d_pretrain",
        action="store_true",
        help="Initialize the 3D MAE encoder from the mapped 2D model checkpoint.",
    )
    parser.add_argument(
        "--pretrained_2d_ckpt_dir",
        default=DEFAULT_2D_CKPT_DIR,
        help="Directory containing CONCH/UNI/H-optimus-1 checkpoints.",
    )
    parser.add_argument(
        "--pretrained_2d_ckpt",
        default="",
        help="Optional explicit 2D checkpoint path. Auto mapping is used when empty.",
    )
    parser.add_argument(
        "--pretrained_2d_source",
        default="auto",
        choices=["auto", "CONCH", "UNI", "H-optimus-1"],
        help="2D checkpoint source label. Auto maps B/L/H to CONCH/UNI/H-optimus-1.",
    )
    return parser


def main(args):
    misc.init_distributed_mode(args)

    if args.decoder_arch == "vit-small":
        args.decoder_embed_dim = 384
        args.decoder_depth = 12
        args.decoder_num_heads = 6

    if args.data_source == "prostate":
        chunk_size = infer_chunk_size(args.prostate_patch_config)
        uses_patch14 = args.model == "mae_vit_huge_patch14"
        target_size = 224 if uses_patch14 else chunk_size
        if args.input_size == 224:
            args.input_size = target_size
        if args.num_frames == 16:
            args.num_frames = target_size
        if args.t_patch_size == 2:
            args.t_patch_size = 16
        if args.pred_t_dim == 8:
            args.pred_t_dim = args.num_frames

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for CUDA matmul/cuDNN kernels")

    if args.data_source == "prostate":
        train_cohorts = resolve_loco_cohorts(
            args.prostate_cohorts, args.leave_one_cohort
        )
        print(
            "Prostate pretraining cohorts: {} | held out: {} | cache: {}".format(
                train_cohorts,
                args.leave_one_cohort or "none",
                args.prostate_patch_config,
            )
        )
        dataset_train = ProstatePatchDataset(
            metadata_cache_dir=args.metadata_cache_dir,
            cohorts=train_cohorts,
            cache_suffix=args.prostate_patch_config,
            input_size=args.input_size,
            num_frames=args.num_frames,
            normalize=not args.prostate_no_normalize,
        )
        print(
            "Constructed prostate dataset with {} patches from {} slides".format(
                len(dataset_train), dataset_train.num_slides
            )
        )
    else:
        dataset_train = Kinetics(
            mode="pretrain",
            path_to_data_dir=args.path_to_data_dir,
            sampling_rate=args.sampling_rate,
            num_frames=args.num_frames,
            train_jitter_scales=(256, 320),
            repeat_aug=args.repeat_aug,
            jitter_aspect_relative=args.jitter_aspect_relative,
            jitter_scales_relative=args.jitter_scales_relative,
        )
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model_kwargs = vars(args).copy()
    model_kwargs["img_size"] = args.input_size
    model = models_mae.__dict__[args.model](
        **model_kwargs,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.use_2d_pretrain and not args.resume:
        load_2d_pretrained_weights(
            model_without_ddp,
            args.model,
            ckpt_dir=args.pretrained_2d_ckpt_dir,
            ckpt_path=args.pretrained_2d_ckpt,
            source=args.pretrained_2d_source,
        )
    elif args.use_2d_pretrain and args.resume:
        print("=> Skipping 2D pretrain initialization because --resume is set")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    adamw_cls = getattr(getattr(torch.optim, "_multi_tensor", None), "AdamW", torch.optim.AdamW)
    optimizer = adamw_cls(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
        )
        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)

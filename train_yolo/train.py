import datetime
import sys
import os
from torch import distributed as dist

from basicsr.train import load_resume_state, init_tb_loggers, create_train_val_dataloader
from yolov12.ultralytics.utils import RANK, callbacks
from yolov12.ultralytics.utils.checks import check_amp
from yolov12.ultralytics.utils.torch_utils import autocast, TORCH_2_4

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import logging
import math
import time
import torch
from torch import nn
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

from yolov12.ultralytics.models.yolo import YOLO

def setup_mambair():
    # MambaIR code
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None and opt['rank'] == 0:
      make_exp_dirs(opt)
      if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
        mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    mambaIR_model = build_model(opt)

    if resume_state:  # resume training
      mambaIR_model.resume_training(resume_state)  # handle optimizers and schedulers
      logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
      start_epoch = resume_state['epoch']
      current_iter = resume_state['iter']
    else:
      start_epoch = 0
      current_iter = 0

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")





# YOLO code



# #------------------
# # Forward
# batch = 'xy'
# tloss = None
# i = 42
#
# with autocast(model.amp):
#   batch = model.preprocess_batch(batch)
#   loss,loss_items = model(batch)
#   tloss = (
#     (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
#   )
#
# # Backward
# model.scaler.scale(loss).backward()
# #------------------


def train_yolo():
    from yolov12.ultralytics.cfg import TASK2DATA
    from yolov12.ultralytics.utils import yaml_load, checks, DEFAULT_CFG_DICT

    model = YOLO('yolov12n.yaml')

    results = model.train(
        data='/home/louis/workspace/MambaIR/train_yolo/data-dota-4x.yaml',
        epochs=2,
        batch=8,
        imgsz=1024,
        scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
        mosaic=1.0,
        mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
        copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
        device="0",
    )

    # if isinstance(model.args.device, str) and len(model.args.device):  # i.e. device='0' or device='0,1,2,3'
    #     world_size = len(model.args.device.split(","))
    # elif isinstance(model.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
    #     world_size = len(model.args.device)
    # elif model.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
    #     world_size = 0
    # elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
    #     world_size = 1  # default to device 0
    # else:  # i.e. device=None or device=''
    #     world_size = 0
    world_size = 1

    overrides = model.overrides
    custom = {
        # NOTE: handle the case when 'cfg' includes 'data'.
        "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[model.task],
        "model": overrides["model"],
        "task": model.task,
    }  # method defaults
    args = {**overrides, **custom, "mode": "train"}  # highest priority args on the right
    if args.get("resume"):
        args["resume"] = model.ckpt_path
    if not args.get("resume"):  # manually set model only if not resuming
        model.trainer.model = model.trainer.get_model(weights=model.model if model.ckpt else None, cfg=model.model.yaml)
        model.model = model.trainer.model

    # Trainer code here
    model.model = model.model.to(model.device)

    # set attributes
    model.model.nc = model.data["nc"]  # attach number of classes to model
    model.model.names = model.data["names"]  # attach class names to model
    model.model.args = model.args  # attach hyperparameters to model

    model.trainer._setup_train(model.trainer, world_size)

    batch = ''
    i = 'iteration'
    tloss = 1

    with autocast(model.amp):
        batch = model.preprocess_batch(batch)
        loss, loss_items = model.model(batch)
        if RANK != -1:
            loss *= world_size
        tloss = (
            (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
        )

    # Freeze layers
    # freeze_list = (
    #     model.args.freeze
    #     if isinstance(model.args.freeze, list)
    #     else range(model.args.freeze)
    #     if isinstance(model.args.freeze, int)
    #     else []
    # )
    # always_freeze_names = [".dfl"]  # always freeze these layers
    # freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
    # for k, v in model.model.named_parameters():
    #     # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
    #     if any(x in k for x in freeze_layer_names):
    #         # LOGGER.info(f"Freezing layer '{k}'")
    #         v.requires_grad = False
    #     elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
    #         # LOGGER.info(
    #         #     f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
    #         #     "See ultralytics.engine.trainer for customization of frozen layers."
    #         # )
    #         v.requires_grad = True
    #
    # model.amp = torch.tensor(args.amp).to(model.device)  # True or False
    # if model.amp and RANK in {-1, 0}:  # Single-GPU and DDP
    #     callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
    #     model.amp = torch.tensor(check_amp(model.model), device=model.device)
    #     callbacks.default_callbacks = callbacks_backup  # restore callbacks
    #
    # if RANK > -1 and world_size > 1:  # DDP
    #     dist.broadcast(model.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
    # model.amp = bool(model.amp)  # as boolean
    # model.scaler = (
    #     torch.amp.GradScaler("cuda", enabled=model.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=model.amp)
    # )
    # if world_size > 1:
    #     model.model = nn.parallel.DistributedDataParallel(model.model, device_ids=[RANK], find_unused_parameters=True)
    #     model.set_model_attributes()  # set again after DDP wrapper

    # Just copied from model, not used so far
    # if RANK in {-1, 0}:
    #     ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
    #     self.model, self.ckpt = attempt_load_one_weight(ckpt)
    #     self.overrides = self.model.args
    #     self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
    # return self.metrics

    # Train the model
    results = model.train(
        data='/home/louis/workspace/MambaIR/train_yolo/data-dota-4x.yaml',
        epochs=2,
        batch=8,
        imgsz=1024,
        scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
        mosaic=1.0,
        mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
        copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
        device="0",
    )

# # Evaluate model performance on the validation set
# metrics = model.val()
#
# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()


def training_loop(start_epoch, total_epochs, train_sampler, prefetcher, total_iters, mambaIR_model, opt, current_iter):
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            mambaIR_model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            mambaIR_model.feed_data(train_data)

if __name__ == '__main__':
    # Train the model
    train_yolo()

    # # Start the training loop
    # training_loop(start_epoch, total_epochs, current_iter)
import datetime
import sys
import os

from basicsr.train import load_resume_state, init_tb_loggers, create_train_val_dataloader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from yolov12.ultralytics import YOLO


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

# YOLO code

model = YOLO('yolov12n.yaml')

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
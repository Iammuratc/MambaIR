import os

import torch
from collections import OrderedDict

from torch import nn
from torch.nn import functional as F, DataParallel
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from basicsr.utils import imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from yolov12.ultralytics.models import YOLO
from yolov12.ultralytics.utils.loss import v8DetectionLoss
from yolov12.ultralytics.utils import IterableSimpleNamespace

from os import path as osp


class CombinedSRYoloNet(torch.nn.Module):
    def __init__(self, sr_net, yolo_net):
        super(CombinedSRYoloNet, self).__init__()
        self.sr_net = sr_net
        self.yolo_net = yolo_net

    def forward(self, x):
        sr_out = self.sr_net(x)
        yolo_out = self.yolo_net(sr_out)
        return sr_out, yolo_out


@MODEL_REGISTRY.register()
class MambaIRv2YoloModel(SRModel):
    """MambaIRv2 model for image restoration + object detection."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.opt = opt
        # define network
        self.net_g = build_network(opt['network_g'])
        # self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # Add yolo model and wrap in nn.Sequential
        self.yolo_model = YOLO('yolov12n.yaml')
        # self.yolo_model = self.yolo_model.to(self.device)
        self.yolo_model.model.args['box'] = opt['yolo_losses']['box_gain']
        self.yolo_model.model.args['cls'] = opt['yolo_losses']['cls_gain']
        self.yolo_model.model.args['dfl'] = opt['yolo_losses']['dlf_gain']
        self.yolo_model.model.args = IterableSimpleNamespace(**self.yolo_model.model.args)
        self.criterion = v8DetectionLoss(self.yolo_model.model)

        self.combined_net = CombinedSRYoloNet(self.net_g, self.yolo_model.model)
        self.combined_net = self.model_to_device(self.combined_net)

        if self.is_train:
            self.init_training_settings()

    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'label' in data:
            self.label = data['label']#.to(self.device)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        sr_out, yolo_out = self.combined_net(self.lq)

        self.output = sr_out
        preds = yolo_out

        yolo_in = {
            'img': self.output.clone(),
            'cls': self.label['cls'],
            'bboxes': self.label['bboxes'],
            'batch_idx': self.label['batch_idx'],
        }

        # Forward pass through the YOLO model
        if len(self.label['cls']) > 0:
            # preds = self.yolo_model.model.forward(self.output)
            yolo_loss = self.criterion(preds, yolo_in)[0]
        else:
            yolo_loss = 0

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total *= self.opt['sr_weight']
        # Additional loss for YOLO model
        l_total += self.opt['yolo_weight'] * yolo_loss

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.combined_net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        print(f"Save image path: {self.opt['path']['visualization']}")

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'images',
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        os.symlink(self.opt['val']['dataroot_label'], osp.join(self.opt['path']['visualization'], dataset_name, 'labels'))

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter):
        if current_iter == -1:
            current_iter = 'latest'
        # Save SR model
        save_filename = f'net_g_{current_iter}.pth'
        sr_save_path = os.path.join(self.opt['path']['models'], save_filename)

        print(f"SR model save path: {sr_save_path}")

        if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
            sr_net = self.net_g.module
        sr_net_state_dict = sr_net.state_dict()
        for key, param in sr_net_state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            sr_net_state_dict[key] = param.cpu()

        save_dict = {}
        save_dict['params'] = sr_net_state_dict
        torch.save(save_dict, sr_save_path)

        # Save Yolo model

        save_filename = f'yolo_{current_iter}.pt'
        yolo_save_path = os.path.join(self.opt['path']['models'], save_filename)

        from copy import deepcopy
        from yolov12.ultralytics import __version__
        from datetime import datetime

        updates = {
            "model": deepcopy(self.yolo_model).half() if isinstance(self.yolo_model, nn.Module) else self.yolo_model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.yolo_model.ckpt, **updates}, yolo_save_path)

        print(f"Yolo model save path: {yolo_save_path}")




        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
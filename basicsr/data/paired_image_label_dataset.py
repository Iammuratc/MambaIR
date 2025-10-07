from pathlib import Path

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, \
     paired_label_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop, augment_imgaug
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

from yolov12.ultralytics.data.utils import verify_image_label

import numpy as np

@DATASET_REGISTRY.register()
class PairedImageLabelDataset(data.Dataset):
    """Paired image dataset for image restoration.
    With labels for subsequent object detection

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageLabelDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder, self.lq_folder, self.label_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_label']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':   # Unused case?
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.label_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'label']
            # self.paths = paired_label_paths_from_lmdb([self.lq_folder, self.gt_folder, self.label_folder], ['lq', 'gt', 'label'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None: # Unused case?
            #TODO?
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:   # Used case?
            self.paths = paired_label_paths_from_folder([self.lq_folder, self.gt_folder, self.label_folder], ['lq', 'gt', 'label'], self.filename_tmpl, self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32., H W 3
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        label_path = str(Path(self.label_folder[0]) / Path(gt_path).name.split('.')[0]) + '.txt'

        prefix = '\x1b[34m\x1b[1mtrain: \x1b[0m'
        keypoint = False
        num_cls = 4
        nkpt = 0
        ndim = 0

        im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label((img_gt, label_path, prefix, keypoint, num_cls, nkpt, ndim), verify_img=False)

        label = {
            'im_file': gt_path,
            'shape': img_gt.shape,
            "cls": lb[:, 0:1],  # n, 1
            "bboxes": lb[:, 1:],  # n, 4
            "segments": segments,
            "keypoints": keypoint,
            "normalized": True,
            "bbox_format": "xywh",
                        }

        # augmentation for training # TODO: augmentations
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, label = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path, label)
            # flip, rotation
            ## Ground truth must be the first one in the list
            img_gt, img_lq, label = augment_imgaug(imgs=[img_gt, img_lq], opt=self.opt['augmentations'], label=label)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {'lq': img_lq, 'gt': img_gt, 'label': label, 'lq_path': lq_path, 'gt_path': gt_path, 'label_path': label_path}

    def __len__(self):
        return len(self.paths)

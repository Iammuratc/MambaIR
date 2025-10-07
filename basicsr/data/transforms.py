import cv2
import random

import numpy as np
import torch

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import traceback

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None, label=None, cutoff_thresh=0.5):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # tfull_img = img_gts[0].copy()

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    if h_gt == gt_patch_size and w_gt == gt_patch_size:
        # If the GT image is already the same size as the patch size,
        # just return the images without cropping
        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        return img_gts, img_lqs, label

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    if label is not None:
        patch_xmin = (left * scale) / w_gt
        patch_ymin = (top * scale) / h_gt
        patch_xmax = (left * scale + gt_patch_size) / w_gt
        patch_ymax = (top * scale + gt_patch_size) / h_gt

        new_bboxes = []
        new_segments = []
        new_classes = []

        # test_patch = img_gts if not type(img_gts) == list else img_gts[0]

        for cls, bbox, seg in zip(label['cls'], label['bboxes'], label['segments']):
            # for (y, x) in seg.tolist():
            #     x_scale = int(x * (w_gt-1))
            #     y_scale = int(y * (h_gt-1))
            #     tfull_img[x_scale, y_scale, 0] = 1
            #     tfull_img[x_scale, y_scale, 1] = 0
            #     tfull_img[x_scale, y_scale, 2] = 0

            cx, cy, bw, bh = bbox
            x1 = cx - bw / 2
            x2 = cx + bw / 2
            y1 = cy - bh / 2
            y2 = cy + bh / 2

            inter_x1 = max(x1, patch_xmin)
            inter_y1 = max(y1, patch_ymin)
            inter_x2 = min(x2, patch_xmax)
            inter_y2 = min(y2, patch_ymax)

            inter_w = max(inter_x2 - inter_x1, 0)
            inter_h = max(inter_y2 - inter_y1, 0)
            inter_area = inter_w * inter_h
            orig_area = bw * bh

            # Skip if overlap is too small
            if inter_area / orig_area < cutoff_thresh:
                continue

            # Shift and clip the segmentation
            seg_np = np.array(seg)
            seg_np[:, 0] = (seg_np[:, 0] - patch_xmin) / (patch_xmax - patch_xmin)
            seg_np[:, 1] = (seg_np[:, 1] - patch_ymin) / (patch_ymax - patch_ymin)

            # Clamp segment points to [0, 1]
            seg_np = np.clip(seg_np, 0, 1)

            # Update bbox to new cropped location
            cx_new = (inter_x1 + inter_x2) / 2 - patch_xmin
            cy_new = (inter_y1 + inter_y2) / 2 - patch_ymin
            bw_new = inter_x2 - inter_x1
            bh_new = inter_y2 - inter_y1

            # Normalize to patch
            cx_new /= (patch_xmax - patch_xmin)
            cy_new /= (patch_ymax - patch_ymin)
            bw_new /= (patch_xmax - patch_xmin)
            bh_new /= (patch_ymax - patch_ymin)

            new_bboxes.append([cx_new, cy_new, bw_new, bh_new])
            new_segments.append(seg_np.tolist())
            new_classes.append(cls)

            # for (y, x) in seg_np.tolist():
            #     x_scale = int(x * (gt_patch_size-1))
            #     y_scale = int(y * (gt_patch_size-1))
            #     test_patch[x_scale, y_scale, 0] = 1
            #     test_patch[x_scale, y_scale, 1] = 0
            #     test_patch[x_scale, y_scale, 2] = 1

        label['bboxes'] = new_bboxes
        label['segments'] = new_segments
        label['cls'] = new_classes

    return img_gts, img_lqs, label


def augment_imgaug(imgs, opt, label=None):
    '''
    Augment images using imgaug library.
    '''
    aug = iaa.Sequential([
        # Horizontal flip with 50% probability
        iaa.Fliplr(opt['fliplr']),
        
        # Vertical flip with 50% probability
        iaa.Flipud(opt['flipud']),
        
        # Rotate by -45 to +45 degrees
        # iaa.Rotate((-opt['rotate'], opt['rotate'])),
        
        # Rotate 90 degrees
        iaa.Rot90((opt['rot90'][0],opt['rot90'][1])),
        # Add per-channel noise
        iaa.Add((-opt['noise_per_channel'], opt['noise_per_channel']), per_channel=True),
        
        # Gaussian blur with random sigma
        # iaa.GaussianBlur(sigma=(0.0, opt['gaussian_sigma']))
    ], random_order=True).to_deterministic()  # Apply augmentations in random order
    
    if label is None:
        # print(imgs[0].shape, imgs[1].shape)
        img_0 = aug(image=imgs[0])
        img_1 = aug(image=imgs[1])
        # print(img_0.shape, img_1.shape)
        return img_0, img_1
    else:
        try:
            h,w = imgs[0].shape[:2]
            bbox_list = []
            for bbox in label['bboxes']:
                (cx, cy, bw, bh) = bbox
                bbox_list.append(BoundingBox(x1=(cx-bw/2)*w, y1=(cy-bh/2)*h, x2=(cx+bw/2)*w, y2=(cy+bh/2)*h))
            bbs = BoundingBoxesOnImage(bbox_list, shape=imgs[0].shape)
            # Define the augmentation sequence

            # Apply augmentations to images, mask, and bounding boxes
            img_gt_aug, bbs_aug = aug(image=imgs[0], bounding_boxes=bbs)

            # 3. Apply same aug to second image (no bboxes)
            img_lq_aug = aug(image=imgs[1])
            # images_aug, bbs_aug = aug(images=imgs, bounding_boxes=bbs)
            bbs_aug = bbs_aug.clip_out_of_image()
            if len(label['bboxes']) != len(bbs_aug.bounding_boxes):
                # print(f"Number of bounding boxes (={len(label['bboxes'])}) should remain the same after augmentation (={len(bbs_aug.bounding_boxes)}).")
                # print("Returning original images before augmentations.")
                img_gt_aug, img_lq_aug = imgs[0], imgs[1]
            else:
                label['bboxes'] = [[(bb.x1 + bb.x2) / (2*w), (bb.y1 + bb.y2) / (2*h), (bb.x2 - bb.x1)/w, (bb.y2 - bb.y1)/h] for bb in bbs_aug.bounding_boxes ]

            return img_gt_aug, img_lq_aug, label
        
        except Exception as e:
            print(f"Error occurred: {e}")
            # or use logging
            traceback.print_exc()

def augment(imgs, opt, labels=None, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.
        labels (dict): Dict of bboxes, segmented images and classes
            Default: None

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = opt['use_hflip'] > random.random()
    vflip = opt['use_vflip'] > random.random()
    rot90 = opt['use_rot'] > random.random()
    local_distortion = opt['add_local_distortion'] > random.random()
    noise = opt['add_noise'] > random.random()
    blur = opt['add_blur'] > random.random()

    def _augment_geometric(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        if local_distortion:
            img = add_local_distortion(img)
        return img

    def _augment_pixels(img):
        if noise:
            img = add_gaussian_noise(img)
        if blur:
            img = add_gaussian_blur(img)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    # Pixel-wise augmentation, e.g., noise, blur
    imgs = [imgs[0],_augment_pixels(imgs[1])]
    # Geometric augmentations, e.g., flip, distort
    imgs = [_augment_geometric(img) for img in imgs]
    ## Assuming that ground truth is the first array in imgs, augment the input image, but not the ground truth
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def add_gaussian_noise(img, mean=0, var=0.01):
    image_float = np.float32(img) / 255.0

    # Generate Gaussian noise
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, image_float.shape)

    # Add the Gaussian noise to the image
    noisy_image = image_float + gaussian_noise

    # Clip the values to keep them in the range [0, 1], and convert back to 8-bit image
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = np.uint8(noisy_image * 255)
    return noisy_image

def add_gaussian_blur(img,kernel_size=(15,15),sigma=0):
    # Define the kernel size and standard deviation for Gaussian blur
    ## sigma: Standard deviation in X and Y direction; 0 lets OpenCV compute it based on kernel size

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(img, kernel_size, sigma)
    return blurred_image

def add_local_distortion(img, coeff_0_max = 1.0, coeff_1_max = 1.0):
    coeff_0 = random.uniform(0,coeff_0_max)
    coeff_1 = random.uniform(0,coeff_1_max)

    # Get the image dimensions
    rows, cols = img.shape[:2]

    # Create a meshgrid for the pixel coordinates
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Normalize coordinates to the range [-1, 1]
    x_norm = (x - cols / 2) / (cols / 2)
    y_norm = (y - rows / 2) / (rows / 2)

    # Calculate the radial distance from the center of the image
    r = np.sqrt(x_norm**2 + y_norm**2)

    # Apply radial distortion based on the distance
    x_distorted = x_norm * (1 + coeff_0 * r**2 + coeff_1 * r**4)
    y_distorted = y_norm * (1 + coeff_0 * r**2 + coeff_1 * r**4)

    # Map distorted coordinates back to pixel space
    x_new = (x_distorted * cols / 2 + cols / 2).astype(np.float32)
    y_new = (y_distorted * rows / 2 + rows / 2).astype(np.float32)

    # Apply remapping
    distorted_image = cv2.remap(img, x_new, y_new, interpolation=cv2.INTER_LINEAR)
    return distorted_image

def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
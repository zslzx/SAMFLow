import sys

# from attr import validate
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from configs.submission import get_cfg as get_submission_cfg
# # from configs.kitti_submission import get_cfg as get_kitti_cfg
# from configs.things_eval import get_cfg as get_things_cfg
# from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
import datasets
from utils import frame_utils

from core.Model import build_flowmodel

from utils.utils import InputPadder, forward_interpolate
import imageio
import itertools
import random

TRAIN_SIZE = [432, 960]

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', division=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // division) + 1) * division - self.ht) % division
        pad_wd = (((self.wd // division) + 1) * division - self.wd) % division
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti436':
            self._pad = [0, 0, 0, 436 - self.ht]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti384':
            self._pad = [0, 0, 0, 384 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

import math
def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

@torch.no_grad()
def inference_with_tile(model, image1, image2, hws, weights, train_size, image_size, convert2nparray=False, padder=None):

    if padder is not None:
        image1, image2 = padder.pad(image1, image2)
    flows = 0
    flow_count = 0
    for idx, (h, w) in enumerate(hws):
        image1_tile = image1[:, :, h:h+train_size[0], w:w+train_size[1]]
        image2_tile = image2[:, :, h:h+train_size[0], w:w+train_size[1]]
        flow_pre, flow_low = model(image1_tile, image2_tile)

        padding = (w, image_size[1]-w-train_size[1], h, image_size[0]-h-train_size[0], 0, 0)
        flows += F.pad(flow_pre * weights[idx], padding)
        flow_count += F.pad(weights[idx], padding)

    flow_pre = flows / flow_count

    if padder is not None:
        flow_pre = padder.unpad(flow_pre)

    if not convert2nparray:
        return flow_pre
    flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

@torch.no_grad()
def create_sintel_submission(model, sigma=0.05, cfg=None):
    """ Create submission for the Sintel leaderboard """

    if cfg is None or cfg.save_path is None:
        assert False, "No save_path"
    output_path = cfg.save_path

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    for dstype in ['final', "clean"]:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
                # break
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()
            flow = inference_with_tile(model, image1, image2, hws, weights, TRAIN_SIZE, IMAGE_SIZE, convert2nparray=True)

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)

@torch.no_grad()
def create_kitti_submission(model, sigma=0.05, cfg=None):
    """ Create submission for the Sintel leaderboard """

    if cfg is None or cfg.save_path is None:
        assert False
    output_path = cfg.save_path

    IMAGE_SIZE = [384, 1242]
    TRAIN_SIZE = [384, 720]

    print(f"output path: {output_path}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"training size: {TRAIN_SIZE}")

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:   # fix the height=432, adaptive ajust the width
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 384
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti384') # padding the image to height of 432
        image1, image2 = image1[None].cuda(), image2[None].cuda()

        flow = inference_with_tile(model, image1, image2, hws, weights, TRAIN_SIZE, IMAGE_SIZE, convert2nparray=True, padder=padder)

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

@torch.no_grad()
def validate_sintel(model, sigma=0.05, cfg=None):
    """ Peform validation using the Sintel (train) split """

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    for dstype in ['final', "clean"]:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)

            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            
            flow_pre = inference_with_tile(model, image1, image2, hws, weights, TRAIN_SIZE, IMAGE_SIZE)
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[f"{dstype}_tile"] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_kitti(model, sigma=0.05, cfg=None):
    IMAGE_SIZE = [432, 1242]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 432
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti432')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow_pre = inference_with_tile(model, image1, image2, hws, weights, TRAIN_SIZE, IMAGE_SIZE, padder=padder)

        flow = flow_pre[0].cpu()
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

import pytorch_lightning as pl

class PLWrap(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_flowmodel(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help='model type: SAMFlow-H | SAMFlow-H-ft | SAMFlow-B | SAMFlow-tiny')
    # parser.add_argument('--model_path', help='ckpt path')
    parser.add_argument('--eval', help='eval benchmark: sintel_validation | kitti_validation | sintel_submission | kitti_submission')
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    if args.model_type == 'SAMFlow-H':
        from configs.SAMFlow_H import get_cfg
        args.model_path = 'weights/SAMFlow-H.ckpt'
    elif args.model_type == 'SAMFlow-H-ft':
        from configs.SAMFlow_H_ft import get_cfg
        args.model_path = 'weights/SAMFlow-H-sintel.ckpt'
    elif args.model_type == 'SAMFlow-B':
        from configs.SAMFlow_B import get_cfg
        args.model_path = 'weights/SAMFlow-B.ckpt'
    elif args.model_type == 'SAMFlow-tiny':
        from configs.SAMFlow_tiny import get_cfg
        args.model_path = 'weights/SAMFlow-tiny.ckpt'
    # elif args.model_type == 'SAMFlow-B':
    #     from configs.things_prompt_submission_basescale import get_cfg
    # elif args.model_type == 'SAMFlow-tiny':
    #     from configs.things_prompt_submission_tiny import get_cfg

    exp_func = None
    cfg = None
    if args.eval == 'sintel_submission':
        exp_func = create_sintel_submission
        cfg = get_cfg()
        cfg.FlowModel.decoder_depth = 32
    elif args.eval == 'kitti_submission':
        exp_func = create_kitti_submission
        cfg = get_cfg()
        cfg.FlowModel.decoder_depth = 24
    elif args.eval == 'sintel_validation':
        exp_func = validate_sintel
        cfg = get_cfg()
        cfg.FlowModel.decoder_depth = 32
    elif args.eval == 'kitti_validation':
        exp_func = validate_kitti
        cfg = get_cfg()
        cfg.FlowModel.decoder_depth = 24
    else:
        print(f"EROOR: {args.eval} is not valid")
    cfg.update(vars(args))

    print(cfg)

    try:
        model = PLWrap.load_from_checkpoint(cfg.model_path, cfg=cfg).model
    except:
        model = torch.nn.DataParallel(build_flowmodel(cfg))
        example_dict_keys = list(model.state_dict().keys())
        model.load_state_dict(torch.load(cfg.model_path))
        model = model.module
    
    model.cuda()
    model.eval()

    exp_func(model, cfg=cfg)

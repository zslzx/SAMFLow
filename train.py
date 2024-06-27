# from __future__ import print_function, division
import numpy as np
import torch
import os
import sys
import argparse
sys.path.append('core')

#import evaluate_FlowFormer_tile as evaluate
import core.datasets as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer

# from core.FlowFormer import FlowFormer
from core.Model import build_flowmodel

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

#torch.autograd.set_detect_anomaly(True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PLWrap(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #print(cfg.freeze_bn)
        self.model = build_flowmodel(cfg)
        if cfg.last_stage_ckpt is not None:
            print("[Loading ckpt from {}]".format(cfg.last_stage_ckpt))
            
            ckpt_dict = torch.load(cfg.last_stage_ckpt, map_location='cpu')

            if 'state_dict' in ckpt_dict:
                ckpt_dict = ckpt_dict['state_dict']
            old_ckpt_dict = self.model.state_dict()
            new_ckpt_dict = {}
            for k in ckpt_dict:
                if k.startswith('module.'):
                    key_in_model = k[7:]
                elif k.startswith('model.'):
                    key_in_model = k[6:]
                else:
                    key_in_model = k
                if key_in_model in old_ckpt_dict and ckpt_dict[k].shape == old_ckpt_dict[key_in_model].shape:
                    new_ckpt_dict[key_in_model] = ckpt_dict[k]
            # quit()
            self.model.load_state_dict(new_ckpt_dict, strict=False)

    def training_step(self, batch, batch_idx):
        image1, image2, flow, valid = [x.cuda() for x in batch]

        if self.cfg.add_noise:
            stdv = np.random.uniform(0.0, 5.0)
            image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
        flow_predictions = self.model(image1, image2)
        loss, metrics = sequence_loss(flow_predictions, flow, valid, self.cfg)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer, scheduler = fetch_optimizer(self.model, self.cfg.trainer)
        scheduler_dict = {
            'scheduler': scheduler,
            # 'interval': 'step',
        }
        return [optimizer], [scheduler_dict]
    
    def train_dataloader(self):
        return datasets.fetch_dataloader(self.cfg)
      




def train(cfg):
    accumulate_grad_batches = cfg.trainer.accumulate_grad_batches
    #cfg.trainer.num_steps *= accumulate_grad_batches #实际上应该不需要
    model = PLWrap(cfg)


    save_dir = os.path.join(cfg.save_root, cfg.model_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = pl_loggers.TensorBoardLogger(save_dir=save_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir+'/ckpt', every_n_train_steps=10000, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(precision='16-mixed', max_steps=cfg.trainer.num_steps, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=1.0, callbacks=[lr_monitor, checkpoint_callback], logger=logger, strategy='ddp_find_unused_parameters_true', sync_batchnorm=True)
    if cfg.resume_path is not None:
        trainer.fit(model, ckpt_path=cfg.resume_path)
    else:
        trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowmodel', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--save_root', type=str)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--last_stage_ckpt', type=str, default=None)
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'things':
        args.model_name = 'SAMFlow_H'
        from configs.train_config.SAMFlow_H_train_things import get_cfg
    elif args.stage == 'sintel':
        args.model_name = 'SAMFlow_H'
        from configs.train_config.SAMFlow_H_train_sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.train_config.SAMFlow_H_train_kitti import get_cfg
    else:
        print('Error: stage is None')
        quit()


    cfg = get_cfg()
    cfg.batch_size = 1
    cfg.update(vars(args))
    cfg.last_stage_ckpt = args.last_stage_ckpt

    torch.manual_seed(1234)
    np.random.seed(1234)

    
    train(cfg)

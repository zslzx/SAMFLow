import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

# from utils.utils import coords_grid, bilinear_sampler, upflow8
from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from ..encoders import twins_svt_large_context, twins_svt_large
from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder, ResidualBlock
from .SAM_encoder import get_encoder, get_encoder_base, get_encoder_tiny
from .CAM import LTSE, ContextAdapter, WeightedAdding

class FlowModel(nn.Module):
    def __init__(self, cfg):
        super(FlowModel, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        if cfg.sam_scale == 'H':
            self.sam_encoder = get_encoder(ft_ckpt=cfg.ft_ver, checkpoint=cfg.sam_checkpoint)
        elif cfg.sam_scale == 'B':
            self.sam_encoder = get_encoder_base(checkpoint=cfg.sam_checkpoint)
        else:
            self.sam_encoder = get_encoder_tiny(checkpoint=cfg.sam_checkpoint)
        self.sam_encoder.requires_grad_(False)

        if cfg.weighted_add:
            self.wadd_1 = WeightedAdding()
        
        self.up_layer8 = nn.Sequential(
            nn.Conv2d(256,96,3,1,1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        
        self.CFM = nn.Sequential(ResidualBlock(256+96, 256), ResidualBlock(256, 256))

        self.LTSE = LTSE(256)
        self.CAM = ContextAdapter(256)

        if self.cfg.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.requires_grad_(False)
                    m.eval()
    
    def train(self, mode):
        super().train(mode)
        self.sam_encoder.eval()
        if self.cfg.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.eval()

    def forward(self, image1, image2):
        with torch.no_grad():
            sam_feat = self.sam_encoder((image1 - self.pixel_mean) / self.pixel_std)
        
        
        sam_feat = self.up_layer8(sam_feat)

        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
        

        context = self.CFM(torch.cat([context, sam_feat], dim=1))
        
        b,_,h,w = image1.shape
        sparse_embedding = self.LTSE((h,w), b)

        context_add = self.CAM(context, sparse_embedding)

        if self.cfg.weighted_add:
            context = self.wadd_1(context, context_add)
        else:
            context = context + context_add
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data)

        return flow_predictions

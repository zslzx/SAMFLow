from typing import Any, Optional, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .SAM.twoway_transformer import TwoWayTransformer

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LTSE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.LTSE_embed = nn.Embedding(1, embed_dim)

    def forward(
        self,
        image_size: Tuple[int, int],
        batch_size: int
    ) -> torch.Tensor:
        return self.LTSE_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

class PositionEmbedding(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None, is_offset: bool = False) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0

        self.positional_encoding_gaussian_matrix = nn.Parameter(scale * torch.randn((2, num_pos_feats)))
        self.is_offset = is_offset
        # self.register_buffer(
        #     "positional_encoding_gaussian_matrix",
        #     scale * torch.randn((2, num_pos_feats)),
        # )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        if not self.is_offset:
            coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class ContextAdapter(nn.Module):
    def __init__(self, transformer_dim):
        super().__init__()
        self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )
        self.flow_init_token = nn.Embedding(2, transformer_dim)
        self.pe_layer = PositionEmbedding(num_pos_feats=transformer_dim//2)
        

    def forward(self, image_embed, sparse_embed):
        '''
            args:
                image_embed: [B,C1,H/8,W/8]
                sparse_embed: [B,C2,D]
        '''

        output_tokens = self.flow_init_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(image_embed.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_embed), dim=1)

        src = image_embed
        b, c, h, w = src.shape
        image_pe = self.pe_layer((h, w)).unsqueeze(0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        hs, src = self.transformer(src, pos_src, tokens)

        src = src.transpose(1, 2).contiguous().view(b, c, h, w)

        return src


class WeightedAdding(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_weight = nn.Parameter(torch.tensor([1,0]).float())
    
    def forward(self, A, B):
        assert A.shape == B.shape

        shapes = A.shape

        # print(A.shape)
        # print(self.add_weight.shape)
        return (torch.cat([A.reshape(1,-1), B.reshape(1,-1)], dim=0) * self.add_weight.reshape(2,1)).sum(dim=0).reshape(*shapes)
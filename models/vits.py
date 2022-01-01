# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_

def test():
  print("Here is the correct vit!")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2) #TODO: Cannot find reference 'functional' in '__init__.py'
        x = self.last_layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads = 1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Now this is cat model
class VisionTransformerCat(nn.Module):
  def __init__(self,
               gene_number=2000,    # Number of genes input into the model
               gene_embed=128,      # Gene embedding dimension
               expression_embed=128,# Expression embeddin gdimension
               depth=1,             # Depth of the model
               heads=1,             # Heads number of this model
               mlp_ratio=4.,        #
               qkv_bias=False,
               qk_scale=None,
               drop_rate=0.,
               attn_drop_rate=0.,
               drop_path_rate=0.1,
               norm_layer=nn.LayerNorm,
               **kwargs):
    super().__init__()
    # Final output dimension is the sum of gene and expression embedding dimensions.
    embed_dim = gene_embed + expression_embed
    self.num_features = self.embed_dim = embed_dim # =3 in this golden truth model

    # class token initiated as -1
    self.cls_token = nn.Parameter(-1 * torch.ones(1,1,embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)
    self.norm = norm_layer(embed_dim)

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])

    # Initiate the positional embedding (Gene embedding)
    self.Embedding = nn.Embedding(gene_number, gene_embed)
    # This line is to adjust if the gene embedding is learnable
    # self.Embedding.weight.requires_grad = False

    self.apply(self._init_weights)

    self.act = nn.LeakyReLU()
    self.exprProj = nn.Linear(1, expression_embed)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def prepare_tokens(self, x):
    B, L = x.shape
    x = self.ReformatInput_cat(x)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    return self.pos_drop(x)

  def ReformatInput_cat(self, x):
    B, G_2 = x.shape
    G = int(G_2/2)
    expr, index = torch.split(x, (G,G), dim = 1)
    geneEmbedding = self.Embedding(index.int())
    expr = expr.reshape(B, G, 1)
    expr -= expr.min(1, keepdim=True)[0]
    expr /= (expr.max(1, keepdim=True)[0]+1e-4)
    #expr = expr.view(B, G, 1) #TODO: What is this for?
    expr = self.exprProj(expr)
    x = torch.cat((expr, geneEmbedding), dim=2)
    #x = expr + geneEmbedding
    return x

  def forward(self, x):
    B, L = x.shape
    x = self.prepare_tokens(x)
    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)
    return x[:,0]

  def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

  def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


# Now this is cat model
class VisionTransformerAdd(nn.Module):
    def __init__(self,
                 gene_number=2000,  # Number of genes input into the model
                 gene_embed=128,  # Gene embedding dimension
                 expression_embed=128,  # Expression embeddin gdimension
                 depth=1,  # Depth of the model
                 heads=1,  # Heads number of this model
                 mlp_ratio=4.,  #
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        # Final output dimension is the sum of gene and expression embedding dimensions.
        # TODO: Check the parameter, if gene embedding dimension is the same with the expression embedding
        embed_dim = gene_embed
        self.num_features = self.embed_dim = embed_dim  # =3 in this golden truth model

        # class token initiated as -1
        self.cls_token = nn.Parameter(-1 * torch.ones(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        # Initiate the positional embedding (Gene embedding)
        self.Embedding = nn.Embedding(gene_number, gene_embed)
        # This line is to adjust if the gene embedding is learnable
        # self.Embedding.weight.requires_grad = False

        self.act = nn.LeakyReLU()
        self.exprProj = nn.Linear(1, expression_embed)
        self.apply(self._init_weights)  # TODO: how?

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, L = x.shape
        x = self.ReformatInput_cat(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.pos_drop(x)

    def ReformatInput_cat(self, x):
        B, G_2 = x.shape
        G = int(G_2 / 2)
        expr, index = torch.split(x, (G, G), dim=1)
        geneEmbedding = self.Embedding(index.int())
        expr = expr.reshape(B, G, 1)
        expr -= expr.min(1, keepdim=True)[0]
        expr /= (expr.max(1, keepdim=True)[0] + 1e-4)
        # expr = expr.view(B, G, 1) #TODO: What is this for?
        expr = self.exprProj(expr)
        x = expr + geneEmbedding
        return x

    def forward(self, x):
        B, L = x.shape
        print(f'The Crop size is {L}!')
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


# Wrapper functions
def vit_cat(gene_number=2000, gene_embed=128, expression_embed=128, depth=3, heads=8, **kwargs):
    model = VisionTransformerCat(
             gene_number=gene_number,
             gene_embed=gene_embed,
             expression_embed=expression_embed,
             depth=depth,
             heads=heads,
             mlp_ratio=4,
             qkv_bias=True,
             norm_layer=partial(nn.LayerNorm, eps=1e-6),
             **kwargs)
    return model

def vit_add(gene_number=2000, gene_embed=128, expression_embed=128, depth=3, heads=8, **kwargs):
    model = VisionTransformerAdd(
            gene_number=gene_number,
            gene_embed=gene_embed,
            expression_embed=expression_embed,
            depth=depth,
            heads=heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs)
    return model


    # def vit_tiny(gene_number = 2000, embed_dim = 128, patch_size=16, **kwargs):
#     model = VisionTransformer(
#         gene_number = gene_number,
#         patch_size=patch_size, embed_dim= embed_dim, depth=3, num_heads=1, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
# def vit_small(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# def vit_base(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

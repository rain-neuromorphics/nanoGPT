from torch import Tensor, autograd
import torch.nn as nn
from typing import Tuple, Callable, Dict, Optional
import torch
from model import LayerNorm, CausalSelfAttention, MLP
from torch.nn import functional as F
import math


class ZOCausalSelfAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        hparams = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        params = {n: p for n, p in self.named_parameters()}
        return ZOCausalSelfAttentionGrad.apply(x, params, hparams)
        
class ZOCausalSelfAttentionGrad(autograd.Function):
    @staticmethod
    def forward(ctx, input, params: Dict, hparams: Dict):
        #TOMYSELF: passer les paramètres en arguments explicites puis construire le dictionnaire à l'intérieur
        return csa(input, params, hparams)
        
    @staticmethod
    def backward(ctx, grad_output):
        ...    

def csa(x: Tensor, params: Dict, hparams: Dict):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    
    c_attn_weight = params['c_attn.weight']
    c_attn_bias = params['c_attn.bias'] if 'c_attn.bias' in params else None
    c_proj_weight = params['c_proj.weight']
    c_proj_bias = params['c_proj.bias'] if 'c_attn.bias' in params else None    

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v  = F.linear(x, c_attn_weight, bias=c_attn_bias).split(hparams['n_embd'], dim=2)
    k = k.view(B, T, hparams['n_head'], C // hparams['n_head']).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, hparams['n_head'], C // hparams['n_head']).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, hparams['n_head'], C // hparams['n_head']).transpose(1, 2) # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    if hparams['flash']:
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=hparams['dropout'] if hparams['training'] else 0, is_causal=True
        )
    else:
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(hparams['bias'][:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=hparams['dropout'], training=hparams['training'])
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = F.dropout(F.linear(y, c_proj_weight, bias=c_proj_bias), p=hparams['dropout'], training=hparams['training'])
    return y

class ZOLayerNorm(LayerNorm):
    def __init__(self, ndim, bias):
        super().__init__(ndim, bias)

    def forward(self, input):
        params = {n: p for n, p in self.named_parameters()}
        ZOLayerNormGrad.apply(input, params)


def layernorm(x: Tensor, params: Dict, hparams: Optional[Dict] = None) -> Tensor:
    return F.layer_norm(x, params['weight'].shape, params['weight'], params['bias'], 1e-5)


class ZOLayerNormGrad(autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, params: Dict):
        return layernorm(input, params)

    @staticmethod
    def backward(ctx, grad_output):
        func = lambda x: F.layer_norm(x, )



def ZO(func: Callable, input: Tensor, params: Dict, hparams: Optional[Dict] = None):
    

    def weight_perturbation(func: Callable, x: Tensor, params: Dict, loss_fn: Callable, eps: float = 1e-4) -> None:
            # create a dictionary of perturbations  
            wps = {n: torch.randn_like(p) for n, p in params.items()}
            
            # apply positive weight perturbation (w + eps * u) and measure loss
            for n, p in params.items():
                p.add_(eps * wps[n])
            loss_p = loss_fn(func(x, params, hparams)) # assume input buffer was built during a forward pass with a forward hook
            
            # apply negative weight perturbation (w - eps * u = w + eps * u - 2 eps * u) and measure loss
            for n, p in params.items():
                p.add_(-2 * eps * wps[n])
                
            loss_n = loss_fn(func(x, params, hparams)) # assume input buffer was built during a forward pass with a forward hook    
            
            # reset weights to their initial value by adding eps * u (w - eps * u + eps * u = w)
            for n, p in params.items():
                p.add_(wps[n])
            
            # manually populate parameter gradients
            for n, p in params.items():
                p.grad = (1 / (2 * eps)) * (loss_p - loss_n) * wps[n]



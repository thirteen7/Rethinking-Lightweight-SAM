from __future__ import annotations

import math

import torch

from torch import nn

from torch.nn import functional as F

from .exact_decoder import (
    attention_forward, heads, operation,
)

def center(value):
    return value - value.mean(dim=-1, keepdim=True)

def native_norm(value, norm):
    output, mean, rstd = torch.native_layer_norm(
        value, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    return output, mean, rstd

def scale_add_first(value, common, row_scale, constant):
    return value.add_(common).mul_(row_scale).add_(constant)

def scale_add_second(value, common, row_scale, constant):
    # One addcmul kernel for the row-scaled base; every term stays FP32.
    return value.addcmul_(row_scale, common).add_(constant)

def factors(attention, image_q, sparse_k, sparse_v, recorder=None, prefix="i2t"):
    """Exact-real full multihead output A @ B + b; r = heads * tokens."""
    q = image_q
    k = operation(recorder, prefix + ".sparse_k", lambda: heads(attention, attention.k_proj(sparse_k)))
    v = operation(recorder, prefix + ".sparse_v", lambda: heads(attention, attention.v_proj(sparse_v)))
    probabilities = operation(recorder, prefix + ".logits_softmax", lambda: torch.softmax(
        (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]), dim=-1))
    count, nheads, image_tokens, sparse_tokens = probabilities.shape
    weight = attention.out_proj.weight.reshape(attention.embedding_dim, nheads, v.shape[-1]).permute(1, 2, 0)
    projected = operation(recorder, prefix + ".sparse_output_projection", lambda: torch.matmul(v, weight))
    A = operation(recorder, prefix + ".factor_layout", lambda: probabilities.transpose(1, 2).reshape(
        count, image_tokens, nheads * sparse_tokens))
    B = projected.reshape(count, nheads * sparse_tokens, attention.embedding_dim)
    return A, B

from __future__ import annotations

import math

from collections import defaultdict

import torch

from torch import nn

def operation(recorder, name, function):
    return function() if recorder is None else recorder.run(name, function)

def heads(attention, value):
    return attention._separate_heads(value, attention.num_heads)

def reassociated_output(attention, probabilities, values, recorder=None, prefix="i2t"):
    """sum_h A_h (V_h W_o,h^T) + b_o, using one dense expansion.

    projected_values: B,H,S,C. expansion_weights: B,N,H*S.
    No B,H,N,C temporary is constructed. out_proj bias is added exactly once.
    """
    batch, count_heads, count_image, count_sparse = probabilities.shape
    channels = attention.embedding_dim
    head_dim = values.shape[-1]
    weight = attention.out_proj.weight.reshape(channels, count_heads, head_dim)
    weight = weight.permute(1, 2, 0)
    projected = operation(recorder, prefix + ".sparse_value_output_projection",
                          lambda: torch.matmul(values, weight))
    factors = operation(recorder, prefix + ".factor_layout",
                        lambda: probabilities.transpose(1, 2).reshape(batch, count_image,
                                                                     count_heads * count_sparse))
    projected = projected.reshape(batch, count_heads * count_sparse, channels)
    output = operation(recorder, prefix + ".dense_factor_expansion",
                       lambda: factors @ projected)
    if attention.out_proj.bias is not None:
        output = operation(recorder, prefix + ".output_bias",
                           lambda: output + attention.out_proj.bias)
    return output

def attention_forward(attention, q, k, v, *, reassociate=False, cached=None,
                      recorder=None, prefix="attention"):
    q_projected = cached.get("q") if cached else None
    k_projected = cached.get("k") if cached else None
    v_projected = cached.get("v") if cached else None
    if q_projected is None:
        q_projected = operation(recorder, prefix + ".q_projection",
                                lambda: heads(attention, attention.q_proj(q)))
    if k_projected is None:
        k_projected = operation(recorder, prefix + ".k_projection",
                                lambda: heads(attention, attention.k_proj(k)))
    if v_projected is None:
        v_projected = operation(recorder, prefix + ".v_projection",
                                lambda: heads(attention, attention.v_proj(v)))
    probabilities = operation(recorder, prefix + ".logits_and_softmax",
                              lambda: torch.softmax((q_projected @ k_projected.transpose(-2, -1)) /
                                                    math.sqrt(q_projected.shape[-1]), dim=-1))
    if reassociate:
        return reassociated_output(attention, probabilities, v_projected, recorder, prefix)
    value = operation(recorder, prefix + ".attention_value_product",
                      lambda: probabilities @ v_projected)
    value = operation(recorder, prefix + ".head_layout",
                      lambda: attention._recombine_heads(value))
    return operation(recorder, prefix + ".dense_output_projection",
                     lambda: attention.out_proj(value))

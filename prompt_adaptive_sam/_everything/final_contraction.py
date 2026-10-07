"""One native PyTorch plan over the complete per-point rank113 representation.

Only shared static K/addK/V are head packed. Dynamic final dense K/V are absent.
Every fold, allocation, projection, scale, reduction and layout is recorded work.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .exact_decoder import heads, operation
from .quadratic_decoder import save, layout


def pack_shared(projected, additive, heads_count=8, recorder=None):
    assert projected.shape == additive.shape and projected.shape[0] == 1
    assert projected.shape[-1] == 320 and heads_count == 8
    rows, width, dim = projected.shape[1], 128, 16
    def keypart(value):
        return value[0].reshape(rows, heads_count, dim).permute(1, 2, 0).contiguous()
    def valuepart(value):
        return value[0].reshape(rows, heads_count, dim).permute(1, 0, 2).contiguous()
    common_k, common_v, common_phase = projected.split((width, width, 64), -1)
    add_k, add_v, add_phase = additive.split((width, width, 64), -1)
    # This assertion and the copy/layout work remain within measured prepare.
    assert torch.equal(add_v, add_v[:, :1].expand_as(add_v))
    return dict(
        Kbar=operation(recorder, "cache.final.Kbar_head_layout", lambda: keypart(common_k)),
        Ak=operation(recorder, "cache.final.Ak_head_layout", lambda: keypart(add_k)),
        Vbar=operation(recorder, "cache.final.Vbar_head_layout", lambda: valuepart(common_v)),
        av=operation(recorder, "cache.final.av_head_view", lambda: add_v[0, 0].reshape(1, heads_count, 1, dim)),
        phase_common=common_phase, phase_add=add_phase,
    )


def factor_values(P, scale, U, Vv, shared, recorder=None, trace=None, audit=None):
    """P*s @ sharedV + (P@U) @ ownV + actual_sum(P)*constant_bias."""
    batch, count_heads, tokens, rows = P.shape
    rank, dim = U.shape[-1], Vv.shape[-1]
    assert count_heads == 8 and tokens == 7 and rank == 113 and dim == 16
    assert U.shape == (batch, rows, rank) and Vv.shape == (batch, count_heads, rank, dim)
    Ps = operation(recorder, "final_factor.P_times_scale", lambda: P * scale[:, None].transpose(-2, -1))
    folded = operation(recorder, "final_factor.Ps_head_query_fold", lambda:
                       Ps.permute(1, 0, 2, 3).reshape(count_heads, batch*tokens, rows))
    common_y = operation(recorder, "final_factor.shared_value_bmm", lambda:
                         torch.bmm(folded, shared["Vbar"]).reshape(count_heads, batch, tokens, dim).permute(1, 0, 2, 3))
    folded_P = operation(recorder, "final_factor.P_head_token56_fold", lambda: P.reshape(batch, count_heads*tokens, rows))
    T = operation(recorder, "final_factor.P_U_no_scale_bmm", lambda: torch.bmm(folded_P, U))
    T_heads = operation(recorder, "final_factor.T_head_view", lambda: T.reshape(batch, count_heads, tokens, rank))
    own_y = operation(recorder, "final_factor.own_value_product", lambda: torch.matmul(T_heads, Vv))
    mass = operation(recorder, "final_factor.actual_sumP", lambda: P.sum(-1, keepdim=True))
    bias_y = operation(recorder, "final_factor.actual_mass_times_av", lambda: mass * shared["av"])
    value = operation(recorder, "final_factor.value_add_fresh_output", lambda: common_y + own_y)
    operation(recorder, "final_factor.value_add_bias", lambda: value.add_(bias_y))
    for name, x in dict(Ps=Ps, P_U=T, actual_sumP=mass, shared_value=common_y,
                        own_value=own_y, bias_value=bias_y, value_heads=value).items():
        save(trace, "final_factor."+name, x)
    if audit is not None:
        audit["final_factor.values"] = dict(P=layout(P), Ps_fold=layout(folded),
                                             P56_fold=layout(folded_P), P_U=layout(T),
                                             output=layout(value), dynamic_KV_materialized=False)
    return value


def factor_attention(attention, query, U, V, scale, shared, projection_weight,
                     recorder=None, trace=None, audit=None):
    batch, rows, rank = U.shape
    assert rank == 113 and V.shape == (batch, rank, 256) and scale.shape == (batch, rows, 1)
    assert query.shape == (batch, 7, 256) and attention.num_heads == 8 and attention.internal_dim == 128
    q = operation(recorder, "final_factor.q_projection", lambda: heads(attention, attention.q_proj(query)))
    kvsmall = operation(recorder, "final_factor.own_V_KV_projection", lambda: F.linear(V, projection_weight[:256]))
    vk, vv = kvsmall.split(128, -1)
    Vk = operation(recorder, "final_factor.Vk_head_view", lambda: vk.reshape(batch, rank, 8, 16).permute(0, 2, 3, 1))
    Vv = operation(recorder, "final_factor.Vv_head_view", lambda: vv.reshape(batch, rank, 8, 16).permute(0, 2, 1, 3))
    qfold = operation(recorder, "final_factor.q_head_query_fold", lambda: q.permute(1, 0, 2, 3).reshape(8, batch*7, 16))
    common_dot = operation(recorder, "final_factor.shared_K_dot", lambda:
                           torch.bmm(qfold, shared["Kbar"]).reshape(8, batch, 7, rows).permute(1, 0, 2, 3))
    add_dot = operation(recorder, "final_factor.shared_addK_dot", lambda:
                        torch.bmm(qfold, shared["Ak"]).reshape(8, batch, 7, rows).permute(1, 0, 2, 3))
    g = operation(recorder, "final_factor.query_Vk_dot", lambda: torch.matmul(q, Vk))
    gfold = operation(recorder, "final_factor.g_head_token56_fold", lambda: g.reshape(batch, 56, rank))
    own_dot = operation(recorder, "final_factor.g_U_no_scale_bmm", lambda:
                        torch.bmm(gfold, U.transpose(1, 2)).reshape(batch, 8, 7, rows))
    logits = operation(recorder, "final_factor.fresh_contiguous_scores", lambda:
                       torch.empty((batch, 8, 7, rows), dtype=U.dtype, device=U.device))
    operation(recorder, "final_factor.shared_dot_times_scale", lambda:
              torch.mul(common_dot, scale[:, None].transpose(-2, -1), out=logits))
    operation(recorder, "final_factor.add_own_no_scale", lambda: logits.add_(own_dot))
    operation(recorder, "final_factor.add_positioned_bias_dot", lambda: logits.add_(add_dot))
    operation(recorder, "final_factor.divide_sqrt16", lambda: logits.div_(math.sqrt(16)))
    P = operation(recorder, "final_factor.native_softmax_all_positions", lambda: torch.softmax(logits, -1))
    value = factor_values(P, scale, U, Vv, shared, recorder, trace, audit)
    combined = operation(recorder, "final_factor.head_recombine", lambda: attention._recombine_heads(value))
    result = operation(recorder, "final_factor.original_out_proj", lambda: attention.out_proj(combined))
    for name, x in dict(q=q, Vk=Vk, Vv=Vv, g=g, common_dot=common_dot, add_dot=add_dot,
                        own_dot=own_dot, logits=logits, P=P, update=result, query_input=query).items():
        save(trace, "final_factor."+name, x)
    if audit is not None:
        audit["final_factor.logits"] = dict(q_fold=layout(qfold), g56_fold=layout(gfold),
                                             own_V_projection=layout(kvsmall), output=layout(logits),
                                             dynamic_KV_materialized=False, rank=rank)
    return result

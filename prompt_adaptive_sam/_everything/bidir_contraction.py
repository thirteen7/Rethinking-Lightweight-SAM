"""Complete own rank56 contractions in both second-layer directions.

Only per-image static Q/K/V/additive terms are packed by head. U0 stays B,N,56:
it is never repeated over heads. Original softmax axes and actual probability
mass survive; no dynamic dense Q/K/V or 384-channel map is built here.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .exact_decoder import heads, operation
from .quadratic_decoder import save, layout


def pack_second_shared(projected, additive, recorder=None):
    assert projected.shape == additive.shape and projected.shape[0] == 1
    assert projected.shape[-1] == 384
    rows = projected.shape[1]
    ck, cv, cq = projected.split(128, -1)
    ak, av, aq = additive.split(128, -1)
    assert torch.equal(av, av[:, :1].expand_as(av))
    def keypart(x):
        return x[0].reshape(rows, 8, 16).permute(1, 2, 0).contiguous()
    def rowpart(x):
        return x[0].reshape(rows, 8, 16).permute(1, 0, 2).contiguous()
    return dict(
        Kbar=operation(recorder, "cache.second.Kbar_head_layout", lambda: keypart(ck)),
        Ak=operation(recorder, "cache.second.Ak_head_layout", lambda: keypart(ak)),
        Vbar=operation(recorder, "cache.second.Vbar_head_layout", lambda: rowpart(cv)),
        av=operation(recorder, "cache.second.av_head_view", lambda: av[0, 0].reshape(1, 8, 1, 16)),
        Qbar=operation(recorder, "cache.second.Qbar_head_layout", lambda: rowpart(cq)),
        Aq=operation(recorder, "cache.second.Aq_head_layout", lambda: rowpart(aq)),
    )


def token_image_values(P, d1, U0, Vv, shared, recorder=None, trace=None, audit=None):
    batch, nh, ns, rows = P.shape
    assert nh == 8 and ns == 7 and U0.shape == (batch, rows, 56)
    assert Vv.shape == (batch, 8, 56, 16) and d1.shape == (batch, rows, 1)
    Pd = operation(recorder, "second_t2i.P_times_d1", lambda: P * d1[:, None].transpose(-2, -1))
    folded = operation(recorder, "second_t2i.Pd_head_query_fold", lambda:
                       Pd.permute(1, 0, 2, 3).reshape(8, batch*7, rows))
    common = operation(recorder, "second_t2i.shared_value_bmm", lambda:
                       torch.bmm(folded, shared["Vbar"]).reshape(8, batch, 7, 16).permute(1, 0, 2, 3))
    Pfold = operation(recorder, "second_t2i.P_head_token56_fold", lambda: P.reshape(batch, 56, rows))
    PU = operation(recorder, "second_t2i.P_U0_no_scale_bmm", lambda: torch.bmm(Pfold, U0))
    PUheads = operation(recorder, "second_t2i.PU0_head_view", lambda: PU.reshape(batch, 8, 7, 56))
    own = operation(recorder, "second_t2i.own_value_product", lambda: torch.matmul(PUheads, Vv))
    mass = operation(recorder, "second_t2i.actual_sumP", lambda: P.sum(-1, keepdim=True))
    bias = operation(recorder, "second_t2i.actual_mass_times_av", lambda: mass * shared["av"])
    value = operation(recorder, "second_t2i.value_fresh_output", lambda: common + own)
    operation(recorder, "second_t2i.value_add_bias", lambda: value.add_(bias))
    for name, x in dict(Pd=Pd, P_U0=PU, actual_sumP=mass, shared_value=common,
                        own_value=own, bias_value=bias, value_heads=value).items():
        save(trace, "second_t2i."+name, x)
    if audit is not None:
        audit["second_t2i.values"] = dict(U0=layout(U0), Pd_fold=layout(folded),
            P56_fold=layout(Pfold), P_U0=layout(PU), output=layout(value),
            U0_repeated_over_heads=False, dynamic_KV_materialized=False)
    return value


def token_image_attention(attention, query, U0, small, d1, shared,
                          recorder=None, trace=None, audit=None):
    batch, rows, rank = U0.shape
    assert rank == 56 and small.shape == (batch, 56, 384) and query.shape == (batch, 7, 256)
    assert d1.shape == (batch, rows, 1) and attention.num_heads == 8 and attention.internal_dim == 128
    q = operation(recorder, "second_t2i.q_projection", lambda: heads(attention, attention.q_proj(query)))
    vk, vv, _ = small.split(128, -1)
    Vk = operation(recorder, "second_t2i.Vk_head_view", lambda: vk.reshape(batch, 56, 8, 16).permute(0, 2, 3, 1))
    Vv = operation(recorder, "second_t2i.Vv_head_view", lambda: vv.reshape(batch, 56, 8, 16).permute(0, 2, 1, 3))
    qfold = operation(recorder, "second_t2i.q_head_query_fold", lambda: q.permute(1, 0, 2, 3).reshape(8, batch*7, 16))
    common = operation(recorder, "second_t2i.shared_K_dot", lambda:
                       torch.bmm(qfold, shared["Kbar"]).reshape(8, batch, 7, rows).permute(1, 0, 2, 3))
    add = operation(recorder, "second_t2i.shared_addK_dot", lambda:
                    torch.bmm(qfold, shared["Ak"]).reshape(8, batch, 7, rows).permute(1, 0, 2, 3))
    g = operation(recorder, "second_t2i.query_Vk_dot", lambda: torch.matmul(q, Vk))
    gfold = operation(recorder, "second_t2i.g_head_token56_fold", lambda: g.reshape(batch, 56, 56))
    own = operation(recorder, "second_t2i.g_U0_no_scale_bmm", lambda:
                    torch.bmm(gfold, U0.transpose(1, 2)).reshape(batch, 8, 7, rows))
    logits = operation(recorder, "second_t2i.fresh_contiguous_logits", lambda:
                       torch.empty((batch, 8, 7, rows), dtype=U0.dtype, device=U0.device))
    operation(recorder, "second_t2i.shared_dot_times_d1", lambda:
              torch.mul(common, d1[:, None].transpose(-2, -1), out=logits))
    operation(recorder, "second_t2i.add_own_no_scale", lambda: logits.add_(own))
    operation(recorder, "second_t2i.add_unscaled_PE_bias", lambda: logits.add_(add))
    operation(recorder, "second_t2i.divide_sqrt16", lambda: logits.div_(math.sqrt(16)))
    P = operation(recorder, "second_t2i.native_softmax_all_image_positions", lambda: torch.softmax(logits, -1))
    value = token_image_values(P, d1, U0, Vv, shared, recorder, trace, audit)
    combined = operation(recorder, "second_t2i.head_recombine", lambda: attention._recombine_heads(value))
    update = operation(recorder, "second_t2i.original_out_proj", lambda: attention.out_proj(combined))
    for name, x in dict(query_input=query, q=q, Vk=Vk, Vv=Vv, g=g, common_dot=common,
                        add_dot=add, own_dot=own, logits=logits, P=P, update=update).items():
        save(trace, "second_t2i."+name, x)
    if audit is not None:
        audit["second_t2i.logits"] = dict(rank=rank, softmax_axis_length=rows, q_fold=layout(qfold),
            g56_fold=layout(gfold), output=layout(logits), dynamic_KV_materialized=False,
            U0_repeated_over_heads=False)
    return update


def image_token_factors(attention, sparse_k, sparse_v, U0, small, d1, shared,
                        recorder=None, trace=None, audit=None):
    batch, rows, rank = U0.shape
    assert rank == 56 and small.shape == (batch, 56, 384)
    assert sparse_k.shape == sparse_v.shape == (batch, 7, 256) and d1.shape == (batch, rows, 1)
    assert attention.num_heads == 8 and attention.internal_dim == 128
    k = operation(recorder, "second_i2t.sparse_k", lambda: heads(attention, attention.k_proj(sparse_k)))
    v = operation(recorder, "second_i2t.sparse_v", lambda: heads(attention, attention.v_proj(sparse_v)))
    kt = operation(recorder, "second_i2t.sparse_k_transpose", lambda: k.transpose(-2, -1))
    kfold = operation(recorder, "second_i2t.k_head_key_fold", lambda: kt.permute(1, 2, 0, 3).reshape(8, 16, batch*7))
    common = operation(recorder, "second_i2t.shared_Q_dot", lambda:
                       torch.bmm(shared["Qbar"], kfold).reshape(8, rows, batch, 7).permute(2, 0, 1, 3))
    add = operation(recorder, "second_i2t.shared_addQ_dot", lambda:
                    torch.bmm(shared["Aq"], kfold).reshape(8, rows, batch, 7).permute(2, 0, 1, 3))
    vq = small[..., 256:]
    Vq = operation(recorder, "second_i2t.Vq_head_view", lambda: vq.reshape(batch, 56, 8, 16).permute(0, 2, 1, 3))
    coefficient = operation(recorder, "second_i2t.Vq_sparseK_dot", lambda: torch.matmul(Vq, kt))
    coeffold = operation(recorder, "second_i2t.coefficient_head_key56_fold", lambda:
                         coefficient.permute(0, 2, 1, 3).reshape(batch, 56, 56))
    own_flat = operation(recorder, "second_i2t.U0_coefficient_no_scale_bmm", lambda: torch.bmm(U0, coeffold))
    own = operation(recorder, "second_i2t.own_logits_head_view", lambda:
                    own_flat.reshape(batch, rows, 8, 7).permute(0, 2, 1, 3))
    logits = operation(recorder, "second_i2t.fresh_contiguous_logits", lambda:
                       torch.empty((batch, 8, rows, 7), dtype=U0.dtype, device=U0.device))
    operation(recorder, "second_i2t.shared_dot_times_d1", lambda: torch.mul(common, d1[:, None], out=logits))
    operation(recorder, "second_i2t.add_own_no_scale", lambda: logits.add_(own))
    operation(recorder, "second_i2t.add_unscaled_PE_bias", lambda: logits.add_(add))
    operation(recorder, "second_i2t.divide_sqrt16", lambda: logits.div_(math.sqrt(16)))
    P = operation(recorder, "second_i2t.native_softmax_all7_sparse_tokens", lambda: torch.softmax(logits, -1))
    weight = operation(recorder, "second_i2t.out_proj_head_weight_view", lambda:
                       attention.out_proj.weight.reshape(256, 8, 16).permute(1, 2, 0))
    projected = operation(recorder, "second_i2t.sparse_output_projection", lambda: torch.matmul(v, weight))
    A = operation(recorder, "second_i2t.factor_A_layout", lambda: P.transpose(1, 2).reshape(batch, rows, 56))
    B = operation(recorder, "second_i2t.factor_B_layout", lambda: projected.reshape(batch, 56, 256))
    for name, x in dict(sparse_k_input=sparse_k, sparse_v_input=sparse_v, k=k, v=v, Vq=Vq,
                        coefficient=coefficient, common_dot=common, add_dot=add, own_dot=own,
                        logits=logits, P=P, A=A, B=B).items():
        save(trace, "second_i2t."+name, x)
    if trace is not None:
        value = P @ v
        save(trace, "second_i2t.value_heads", value)
        save(trace, "second_i2t.update", torch.bmm(A, B) + attention.out_proj.bias)
    if audit is not None:
        audit["second_i2t.logits"] = dict(rank=rank, softmax_axis_length=7, U0=layout(U0),
            k_fold=layout(kfold), coefficient56_fold=layout(coeffold), own_flat=layout(own_flat),
            logits=layout(logits), A=layout(A), B=layout(B),
            dynamic_Q_materialized=False, U0_repeated_over_heads=False,
            full_sparse_keys_and_values=True)
    return A, B

"""E08: full independent moments with writable affine-preloaded baddbmm.

The production moment path never materializes 256-channel norm residuals.
Trace and writable-layout audit are instrumentation, excluded from timing.
No variance clamp, precision change, custom kernel, or prompt state sharing.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .exact_decoder import attention_forward, heads, operation
from .dense_decoder import center, factors, native_norm
from .factor_store import FactorStore


def save(trace, name, value):
    if trace is not None:
        trace[name] = value.detach().cpu().clone()


def layout(value):
    return dict(shape=list(value.shape), stride=list(value.stride()), data_ptr=value.data_ptr(),
                storage_offset=value.storage_offset(), contiguous=value.is_contiguous(),
                dtype=str(value.dtype), device=str(value.device))


def record_write(audit, name, destination, inputs, before_ptr):
    if audit is not None:
        inputs = {k:layout(v) for k,v in inputs.items()}
        assert destination.data_ptr() == before_ptr
        assert all(destination.data_ptr() != v["data_ptr"] for v in inputs.values())
        audit[name] = dict(output=layout(destination), pointer_before=before_ptr,
                          inputs=inputs, fresh_writable_output=True, same_pointer_after_baddbmm=True)


def moment0(S0, s0sq, A0, B0c, recorder=None, audit=None):
    G0 = operation(recorder, "moment0.Gram", lambda: torch.bmm(B0c, B0c.transpose(1, 2)))
    K0 = operation(recorder, "moment0.cross", lambda: torch.matmul(S0, B0c.transpose(1, 2)))
    before_ptr = K0.data_ptr() if audit is not None else None
    Z0 = operation(recorder, "moment0.quadratic_baddbmm_beta2", lambda: K0.baddbmm_(A0, G0, beta=2))
    record_write(audit, "moment0.Z0", Z0, dict(S0=S0,s0sq=s0sq,A0=A0,B0c=B0c,G0=G0), before_ptr)
    return operation(recorder, "moment0.reductions", lambda:
                     (s0sq + (A0 * Z0).sum(-1, keepdim=True)) / S0.shape[-1])


def moment1(F0, fsq, d1, D, W, recorder=None, audit=None):
    G1 = operation(recorder, "moment1.Gram", lambda: torch.bmm(W, W.transpose(1, 2)))
    K1 = operation(recorder, "moment1.cross", lambda: torch.matmul(F0, W.transpose(1, 2)))
    before_ptr = K1.data_ptr() if audit is not None else None
    operation(recorder, "moment1.cross_scale", lambda: K1.mul_(d1))
    Z1 = operation(recorder, "moment1.quadratic_baddbmm_beta2", lambda: K1.baddbmm_(D, G1, beta=2))
    record_write(audit, "moment1.Z1", Z1, dict(F0=F0,fsq=fsq,d1=d1,D=D,W=W,G1=G1), before_ptr)
    return operation(recorder, "moment1.reductions", lambda:
                     (d1.square() * fsq + (D * Z1).sum(-1, keepdim=True)) / F0.shape[-1])


def affine_preload(add, scale, common, left, right, recorder=None, prefix="affine", audit=None):
    # Explicit fresh row-major destination prevents transposed shared F2 from
    # choosing column-major addcmul output and forcing a later layout copy.
    output = operation(recorder, prefix+".writable_contiguous_output", lambda:
                       torch.empty((left.shape[0],left.shape[1],right.shape[2]),
                                   dtype=left.dtype,device=left.device))
    output = operation(recorder, prefix+".addcmul_preload", lambda: torch.addcmul(add, scale, common, out=output))
    before_ptr = output.data_ptr() if audit is not None else None
    output = operation(recorder, prefix+".baddbmm_beta1", lambda: output.baddbmm_(left, right, beta=1))
    record_write(audit, prefix, output, dict(add=add,scale=scale,common=common,left=left,right=right), before_ptr)
    return output


class QuadraticAffineDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.cache = None
        assert not getattr(decoder, "yield_kd_targets", False) and decoder.num_mask_tokens == 4
        first, second = decoder.transformer.layers
        assert len(decoder.transformer.layers) == 2
        assert first.cross_attn_image_to_token.num_heads == second.cross_attn_image_to_token.num_heads == 8
        t2i, i2t = second.cross_attn_token_to_image, second.cross_attn_image_to_token
        final = decoder.transformer.final_attn_token_to_image
        conv1, _, _, conv2, _ = decoder.output_upscaling
        for conv in (conv1, conv2):
            assert conv.kernel_size == conv.stride == (2, 2)
            assert conv.padding == conv.output_padding == (0, 0)
            assert conv.groups == 1 and conv.dilation == (1, 1)
        phase_weight = conv1.weight[:, :, 0, 0].T.contiguous()
        w1 = torch.cat((t2i.k_proj.weight, t2i.v_proj.weight, i2t.q_proj.weight), 0).contiguous()
        b1 = torch.cat((t2i.k_proj.bias, t2i.v_proj.bias, i2t.q_proj.bias), 0)
        pe1 = torch.cat((t2i.k_proj.weight, torch.zeros_like(t2i.v_proj.weight), i2t.q_proj.weight), 0)
        w2 = torch.cat((final.k_proj.weight, final.v_proj.weight, phase_weight), 0).contiguous()
        b2 = torch.cat((final.k_proj.bias, final.v_proj.bias, conv1.bias), 0)
        pe2 = torch.cat((final.k_proj.weight, torch.zeros_like(final.v_proj.weight),
                         torch.zeros_like(phase_weight)), 0)
        phase2 = conv2.weight[:, :, 0, 0].T[:, :, None, None].contiguous()
        for name, value in dict(w1=w1, b1=b1, pe1=pe1, w2=w2, b2=b2, pe2=pe2,
                                phase2=phase2).items():
            self.register_buffer(name, value.detach())
        self.width1, self.width2 = t2i.internal_dim, final.internal_dim
        self.phase_channels = conv1.out_channels
        assert self.w1.shape == (384, 256) and self.w2.shape == (320, 256)

    def clear_cache(self):
        self.cache = None

    @torch.inference_mode()
    def prepare(self, image_embeddings, image_pe, dense_single, recorder=None):
        assert image_embeddings.shape[0] == image_pe.shape[0] == dense_single.shape[0] == 1
        first, second = self.decoder.transformer.layers
        base = operation(recorder, "cache.base", lambda: (image_embeddings + dense_single).flatten(2).transpose(1, 2))
        pe = image_pe.flatten(2).transpose(1, 2)
        positioned = operation(recorder, "cache.positioned", lambda: base + pe)
        t2i, i2t = first.cross_attn_token_to_image, first.cross_attn_image_to_token
        S0 = operation(recorder, "cache.S0", lambda: center(base + i2t.out_proj.bias))
        F1 = operation(recorder, "cache.F1", lambda: S0 * first.norm4.weight)
        F0 = operation(recorder, "cache.F_recenter_after_gamma1", lambda: center(F1))
        F2 = operation(recorder, "cache.F2", lambda: F0 * second.norm4.weight)
        c = operation(recorder, "cache.center_beta1_b1", lambda:
                      center(first.norm4.bias + second.cross_attn_image_to_token.out_proj.bias))
        self.cache = dict(
            embedding=image_embeddings, base=base, pe=pe, S0=S0, F1=F1, F0=F0, F2=F2, c=c,
            s0sq=operation(recorder, "cache.S0_sq", lambda: S0.square().sum(-1, keepdim=True)),
            fsq=operation(recorder, "cache.F_sq", lambda: F0.square().sum(-1, keepdim=True)),
            t2i=dict(k=operation(recorder, "cache.layer0_k", lambda: heads(t2i, t2i.k_proj(positioned))),
                     v=operation(recorder, "cache.layer0_v", lambda: heads(t2i, t2i.v_proj(base)))),
            i2t_q=operation(recorder, "cache.layer0_q", lambda: heads(i2t, i2t.q_proj(positioned))),
            f1_projected=operation(recorder, "cache.F1W", lambda: F.linear(F1, self.w1)),
            f2_projected=operation(recorder, "cache.F2W_phase00", lambda: F.linear(F2, self.w2)),
            add1=operation(recorder, "cache.beta1_bias_PE", lambda:
                           F.linear(pe, self.pe1) + F.linear(first.norm4.bias, self.w1, self.b1)),
            add2=operation(recorder, "cache.beta2_bias_PE", lambda:
                           F.linear(pe, self.pe2) + F.linear(second.norm4.bias, self.w2, self.b2)),
        )
        return self.cache

    @torch.inference_mode()
    def states(self, image_embeddings, image_pe, sparse_prompt_embeddings,
               dense_prompt_embeddings, num_multimask_outputs=3, num_prompts=None,
               recorder=None, trace=None, layout_audit=None):
        assert num_prompts is None and num_multimask_outputs == 3
        assert self.cache is not None and self.cache["embedding"] is image_embeddings
        assert sparse_prompt_embeddings.shape[1] == 2
        decoder, transformer, cache = self.decoder, self.decoder.transformer, self.cache
        first, second = transformer.layers
        count = len(sparse_prompt_embeddings)
        tokens = torch.cat((torch.cat((decoder.iou_token.weight, decoder.mask_tokens.weight))[None].expand(count, -1, -1),
                            sparse_prompt_embeddings), 1)
        assert tokens.shape[1] == 7
        queries = tokens
        own = []
        dense_x = None
        for index, layer in enumerate(transformer.layers):
            prefix = f"layer{index}"
            q = queries if layer.skip_first_layer_pe else queries + tokens
            update = attention_forward(layer.self_attn, q, q, queries, recorder=recorder, prefix=prefix+".self")
            queries = operation(recorder, prefix+".norm1", lambda:
                                layer.norm1(update if layer.skip_first_layer_pe else queries + update))
            t2i, i2t = layer.cross_attn_token_to_image, layer.cross_attn_image_to_token
            if index == 0:
                cached_t2i, image_q = cache["t2i"], cache["i2t_q"]
            else:
                A0, B0, d1, V0 = own[0]
                vw = operation(recorder, "factor1.VW", lambda: F.linear(V0, self.w1))
                d1A0 = operation(recorder, "factor1.d1_A0_reused_by_D", lambda: d1*A0)
                projection = affine_preload(cache["add1"], d1, cache["f1_projected"], d1A0, vw,
                                            recorder, "factor1.affine", layout_audit)
                k, v, q = projection.split(self.width1, -1)
                cached_t2i, image_q = dict(k=heads(t2i, k), v=heads(t2i, v)), heads(i2t, q)
                save(trace, "factor1.projected", projection)
                if trace is not None:
                    x1 = d1 * (cache["F1"] + A0 @ V0) + first.norm4.bias
                    save(trace, "layer0.X", x1)
                    save(trace, "factor1.native_projected_same_state", torch.cat((t2i.k_proj(x1+cache["pe"]),
                         t2i.v_proj(x1), i2t.q_proj(x1+cache["pe"])), -1))
                    del x1
            update = attention_forward(t2i, queries + tokens, None, None, cached=cached_t2i,
                                       recorder=recorder, prefix=prefix+".t2i")
            queries = operation(recorder, prefix+".norm2", lambda: layer.norm2(queries + update))
            update = operation(recorder, prefix+".mlp", lambda: layer.mlp(queries))
            queries = operation(recorder, prefix+".norm3", lambda: layer.norm3(queries + update))
            save(trace, prefix+".sparse_norm3", queries)
            A, B = factors(i2t, image_q, queries + tokens, queries, recorder, prefix+".i2t")
            Bc = operation(recorder, prefix+".center_B", lambda: center(B))
            if index == 0:
                variance = moment0(cache["S0"], cache["s0sq"], A, Bc, recorder, layout_audit)
            else:
                W = operation(recorder, "moment1.W_layout", lambda: torch.cat((center(V0), Bc,
                    cache["c"].reshape(1, 1, -1).expand(count, 1, -1)), 1))
                D = operation(recorder, "moment1.D_layout", lambda:
                              torch.cat((d1A0, A, torch.ones_like(d1)), -1))
                assert D.shape[-1] == W.shape[1] == 113
                variance = moment1(cache["F0"], cache["fsq"], d1, D, W, recorder, layout_audit)
            rstd = operation(recorder, prefix+".rstd_no_clamp", lambda: torch.rsqrt(variance + layer.norm4.eps))
            # Only oracle instrumentation ever constructs the dense residual input.
            if trace is not None:
                before = cache["base"] if index == 0 else dense_x
                y = before + (torch.bmm(A, B) + i2t.out_proj.bias)
                dense_x, mean_dense, rstd_dense = native_norm(y, layer.norm4)
                save(trace, prefix+".Y_dense_own_factors", y)
                save(trace, prefix+".X_dense_own_factors", dense_x)
                save(trace, prefix+".mean_dense_own_factors", mean_dense)
                save(trace, prefix+".var_dense_own_factors", y.var(-1, unbiased=False, keepdim=True))
                save(trace, prefix+".rstd_dense_own_factors", rstd_dense)
                if index == 0:
                    # For LN2 isolation, dense Y uses the candidate's reconstructed X1,
                    # rather than introducing the previous native LN error again.
                    dense_x = rstd * (cache["F1"] + A @ (Bc*layer.norm4.weight)) + layer.norm4.bias
                del y
            save(trace, prefix+".A", A)
            save(trace, prefix+".B", B)
            save(trace, prefix+".variance", variance)
            save(trace, prefix+".rstd", rstd)
            if trace is not None:
                if index == 0:
                    mean = (cache["base"]+i2t.out_proj.bias).mean(-1, keepdim=True) + A @ B.mean(-1, keepdim=True)
                else:
                    mean = d1 * (cache["F1"].mean(-1, keepdim=True) + A0 @ V0.mean(-1, keepdim=True)) + \
                           (first.norm4.bias+i2t.out_proj.bias).mean() + A @ B.mean(-1, keepdim=True)
                save(trace, prefix+".mean_factor", mean)
            V = operation(recorder, prefix+".gamma_B", lambda: Bc * layer.norm4.weight)
            own.append((A, B, rstd, V))
        A0, B0, d1, V0 = own[0]
        A1, B1, d2, V1 = own[1]
        scale = operation(recorder, "factor2.row_scale", lambda: d1*d2)
        U = operation(recorder, "factor2.U_layout", lambda: torch.cat((A0*scale, A1*d2, d2), -1))
        V = operation(recorder, "factor2.V_layout", lambda: torch.cat((center(V0)*second.norm4.weight, V1,
                 (cache["c"]*second.norm4.weight).reshape(1, 1, -1).expand(count, 1, -1)), 1))
        vw = operation(recorder, "factor2.VW_phase00", lambda: F.linear(V, self.w2))
        projection = affine_preload(cache["add2"], scale, cache["f2_projected"], U, vw,
                                    recorder, "factor2.affine_phase00", layout_audit)
        final = transformer.final_attn_token_to_image
        k, v, phase_first = torch.split(projection, (self.width2, self.width2, self.phase_channels), -1)
        save(trace, "factor2.projected", projection)
        if trace is not None:
            x2 = affine_preload(second.norm4.bias,scale,cache["F2"],U,V)
            save(trace, "layer1.X", x2)
            conv = decoder.output_upscaling[0]
            save(trace, "factor2.native_projected_same_state", torch.cat((final.k_proj(x2+cache["pe"]),
                 final.v_proj(x2), F.linear(x2, conv.weight[:, :, 0, 0].T, conv.bias)), -1))
            del x2
        update = attention_forward(final, queries + tokens, None, None,
                                   cached=dict(k=heads(final, k), v=heads(final, v)),
                                   recorder=recorder, prefix="final_t2i")
        queries = operation(recorder, "final_sparse_norm", lambda: transformer.norm_final_attn(queries + update))
        hyper = operation(recorder, "sparse_hyper", lambda: torch.stack(
                 [module(queries[:, 1+i]) for i, module in enumerate(decoder.output_hypernetworks_mlps)], 1))
        head = operation(recorder, "sparse_iou", lambda: decoder.iou_prediction_head(queries[:, 0]))
        save(trace, "final_sparse", queries)
        save(trace, "hyper4", hyper)
        save(trace, "head4", head)
        return dict(U=U, V=V, scale=scale, hyper=hyper, head=head[:, 1:],
                    hw=image_embeddings.shape[-2:]), phase_first

    @torch.inference_mode()
    def reconstruct(self, state, recorder=None, layout_audit=None):
        second = self.decoder.transformer.layers[1]
        x = affine_preload(second.norm4.bias, state["scale"], self.cache["F2"], state["U"], state["V"],
                           recorder, "restore.affine", layout_audit)
        h, w = state["hw"]
        x = x.transpose(1, 2).reshape(len(x), 256, h, w)
        assert x.is_contiguous(memory_format=torch.channels_last), x.stride()
        return x

    @torch.inference_mode()
    def phase(self, phase_first, hyper, hw, all_masks=False, recorder=None):
        _, norm, gelu1, conv2, gelu2 = self.decoder.output_upscaling
        h, w = hw
        feature = phase_first.transpose(1, 2).reshape(len(phase_first), self.phase_channels, h, w)
        feature = operation(recorder, "phase.native_LN2d_GELU", lambda: gelu1(norm(feature)))
        feature = operation(recorder, "phase.second_ConvT00_GELU", lambda:
                            gelu2(F.conv2d(feature, self.phase2, conv2.bias)))
        logits = operation(recorder, "phase.hyper_matmul", lambda: (hyper @ feature.flatten(2)).reshape(-1, 4, h, w))
        return logits if all_masks else logits[:, 1:]

    @torch.inference_mode()
    def local_patch(self, state, cells_yx, all_masks=False, recorder=None):
        h, w = state["hw"]
        ids = torch.arange(len(state["U"]), device=state["U"].device)
        flat = cells_yx[:, 0]*w + cells_yx[:, 1]
        U = operation(recorder, "local.own_factor_row", lambda: state["U"][ids, flat, :][:, None, :])
        scale = operation(recorder, "local.own_row_scale", lambda: state["scale"][ids, flat, :][:, None, :])
        base = operation(recorder, "local.common_base_row", lambda: self.cache["F2"][0, flat, :][:, None, :])
        x = operation(recorder, "local.restore", lambda:
                      torch.bmm(U, state["V"]).addcmul_(scale, base).add_(self.decoder.transformer.layers[1].norm4.bias))
        x = x.transpose(1, 2)[:, :, :, None].contiguous(memory_format=torch.channels_last)
        return self.complete(x, state["hyper"], all_masks=all_masks, recorder=recorder, prefix="local.upscale")

    @torch.inference_mode()
    def complete(self, x2, hyper, all_masks=False, recorder=None, prefix="complete"):
        features = operation(recorder, prefix+".native_upscalers", lambda: self.decoder.output_upscaling(x2))
        masks = operation(recorder, prefix+".hyper_matmul", lambda:
                          (hyper @ features.flatten(2)).reshape(len(x2), 4, *features.shape[-2:]))
        return masks if all_masks else masks[:, 1:]

"""E17 directly consumes own full56 factors in BOTH second-layer directions.

First-layer operations and moments are unchanged. No old states call precedes
the contractions. Final113 attention and native restore/phase/local are inherited.
Dense oracle instrumentation is trace-only and cannot affect production outputs.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .quadratic_decoder import (
    QuadraticAffineDecoder, FactorStore, moment0, moment1, save, affine_preload,
)
from .exact_decoder import attention_forward, heads, operation
from .dense_decoder import center, factors, native_norm
from .final_contraction import pack_shared, factor_attention
from .final_decoder import FactorFinalAttentionDecoder
from .bidir_contraction import (
    pack_second_shared, token_image_attention, image_token_factors,
)


class BidirectionalFactorDecoder(FactorFinalAttentionDecoder):
    @torch.inference_mode()
    def prepare(self, image_embeddings, image_pe, dense_single, recorder=None):
        cache = super().prepare(image_embeddings, image_pe, dense_single, recorder)
        assert torch.count_nonzero(self.pe1[128:256]) == 0
        cache["second_shared"] = pack_second_shared(cache["f1_projected"], cache["add1"], recorder)
        return cache

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
            save(trace, prefix+".sparse_norm1", queries)
            t2i, i2t = layer.cross_attn_token_to_image, layer.cross_attn_image_to_token
            if index == 0:
                cached_t2i, image_q = cache["t2i"], cache["i2t_q"]
            else:
                A0, B0, d1, V0 = own[0]
                vw = operation(recorder, "factor1.VW", lambda: F.linear(V0, self.w1))
                d1A0 = operation(recorder, "factor1.d1_A0_reused_by_D", lambda: d1*A0)
                save(trace, "factor1.U0", d1A0)
                save(trace, "factor1.V0", V0)
                save(trace, "factor1.small_projected", vw)
            if index == 0:
                update = attention_forward(t2i, queries + tokens, None, None, cached=cached_t2i,
                                           recorder=recorder, prefix=prefix+".t2i")
            else:
                update = token_image_attention(t2i, queries + tokens, d1A0, vw, d1,
                         cache["second_shared"], recorder=recorder, trace=trace, audit=layout_audit)
                if trace is not None:
                    x1 = d1 * (cache["F1"] + A0 @ V0) + first.norm4.bias
                    save(trace, "layer0.X", x1)
                    q = heads(t2i, t2i.q_proj(queries+tokens))
                    k = heads(t2i, t2i.k_proj(x1+cache["pe"]))
                    v = heads(t2i, t2i.v_proj(x1))
                    dense_logits = (q @ k.transpose(-2, -1)) / math.sqrt(16)
                    dense_P = torch.softmax(dense_logits, -1)
                    dense_value = dense_P @ v
                    dense_update = t2i.out_proj(t2i._recombine_heads(dense_value))
                    for name, value in dict(logits=dense_logits, P=dense_P,
                                           value_heads=dense_value, update=dense_update).items():
                        save(trace, "second_t2i_dense_same_state."+name, value)
                    del x1, q, k, v, dense_logits, dense_P, dense_value, dense_update
            queries = operation(recorder, prefix+".norm2", lambda: layer.norm2(queries + update))
            save(trace, prefix+".sparse_norm2", queries)
            update = operation(recorder, prefix+".mlp", lambda: layer.mlp(queries))
            queries = operation(recorder, prefix+".norm3", lambda: layer.norm3(queries + update))
            save(trace, prefix+".sparse_norm3", queries)
            if index == 0:
                A, B = factors(i2t, image_q, queries + tokens, queries, recorder, prefix+".i2t")
            else:
                A, B = image_token_factors(i2t, queries + tokens, queries, d1A0, vw, d1,
                        cache["second_shared"], recorder=recorder, trace=trace, audit=layout_audit)
                if trace is not None:
                    x1 = d1 * (cache["F1"] + A0 @ V0) + first.norm4.bias
                    q = heads(i2t, i2t.q_proj(x1+cache["pe"]))
                    k = heads(i2t, i2t.k_proj(queries+tokens))
                    v = heads(i2t, i2t.v_proj(queries))
                    dense_logits = (q @ k.transpose(-2, -1)) / math.sqrt(16)
                    dense_P = torch.softmax(dense_logits, -1)
                    dense_value = dense_P @ v
                    dense_update = i2t.out_proj(i2t._recombine_heads(dense_value))
                    for name, value in dict(logits=dense_logits, P=dense_P,
                                           value_heads=dense_value, update=dense_update).items():
                        save(trace, "second_i2t_dense_same_state."+name, value)
                    del x1, q, k, v, dense_logits, dense_P, dense_value, dense_update
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
        final = transformer.final_attn_token_to_image
        phase_v = operation(recorder, "factor2.V_phase64_projection", lambda: F.linear(V, self.w2[256:]))
        shared = cache["final_shared"]
        phase_first = affine_preload(shared["phase_add"], scale, shared["phase_common"], U, phase_v,
                                     recorder, "factor2.affine_phase64", layout_audit)
        update = factor_attention(final, queries + tokens, U, V, scale, shared, self.w2,
                                  recorder=recorder, trace=trace, audit=layout_audit)
        save(trace, "factor2.phase64", phase_first)
        if trace is not None:
            # Explicit dense oracle is AFTER the production contraction; it is
            # instrumentation only and never contributes to runtime outputs.
            x2 = affine_preload(second.norm4.bias, scale, cache["F2"], U, V)
            save(trace, "layer1.X", x2)
            k = heads(final, final.k_proj(x2 + cache["pe"]))
            v = heads(final, final.v_proj(x2))
            q = heads(final, final.q_proj(queries + tokens))
            dense_logits = (q @ k.transpose(-2, -1)) / math.sqrt(16)
            dense_P = torch.softmax(dense_logits, -1)
            dense_value = dense_P @ v
            dense_update = final.out_proj(final._recombine_heads(dense_value))
            save(trace, "final_dense_same_state.logits", dense_logits)
            save(trace, "final_dense_same_state.P", dense_P)
            save(trace, "final_dense_same_state.value_heads", dense_value)
            save(trace, "final_dense_same_state.update", dense_update)
            save(trace, "factor2.native_phase_same_state", F.linear(x2, self.w2[256:], self.b2[256:]))
            del x2, k, v, q, dense_logits, dense_P, dense_value, dense_update
        queries = operation(recorder, "final_sparse_norm", lambda: transformer.norm_final_attn(queries + update))
        hyper = operation(recorder, "sparse_hyper", lambda: torch.stack(
                 [module(queries[:, 1+i]) for i, module in enumerate(decoder.output_hypernetworks_mlps)], 1))
        head = operation(recorder, "sparse_iou", lambda: decoder.iou_prediction_head(queries[:, 0]))
        save(trace, "final_sparse", queries)
        save(trace, "hyper4", hyper)
        save(trace, "head4", head)
        return dict(U=U, V=V, scale=scale, hyper=hyper, head=head[:, 1:],
                    hw=image_embeddings.shape[-2:]), phase_first

# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch MBART model."""
from functools import partial
import math
import torch.nn.functional as F

import random
from typing import Any, Callable, Optional, Tuple, Union
import torch
from torch import nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.mbart.modeling_mbart import (
    MBartAttention,
    MBartDecoder,
    MBartForCausalLM,
    _expand_mask,
    logger,
)

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available, using PyTorch fallback")
    XFORMERS_AVAILABLE = False
    import torch.nn.functional as F
    
    class DummyXops:
        @staticmethod

        def memory_efficient_attention(
            query, key, value,
            attn_bias=None,
            p: float = 0.0,
            dropout_p: float = None,
            attn_mask=None,
            scale=None,
            head_dim=None,
            training=False,
        ):
            dp = dropout_p if dropout_p is not None else p
          
            # Always use fallback computation for mismatched dimensions
            try:
                scores = query @ key.transpose(-2, -1)
                if scale is not None:
                    scores = scores * scale
                elif head_dim is not None:
                    scores = scores / math.sqrt(head_dim)
                
                # Apply attention bias/mask if provided
                if attn_bias is not None:
                    scores = scores + attn_bias
                elif attn_mask is not None:
                    scores = scores.masked_fill(attn_mask == 0, float('-inf'))
                
                weights = F.softmax(scores, dim=-1)
                if dp > 0 and training:
                    weights = F.dropout(weights, p=dp, training=True)
                
                attn_output = weights @ value
                return attn_output
                
            except Exception as e:
                print(f"Fallback attention computation failed: {e}")
                # Return zeros as last resort with correct shape
                B, H, Q, D = query.shape
                return torch.zeros(B, H, Q, D, device=query.device, dtype=query.dtype)

    xops = DummyXops()


def apply_fast_mbart_decoder(model: MBartForCausalLM) -> None:
    for module in model.model.modules():
        if isinstance(module, MBartDecoder):
            module.forward = partial(mbart_decoder_fast_forward, module)
        if isinstance(module, MBartAttention):
            module.forward = partial(mbart_attention_fast_forward, module)


def mbart_attention_fast_forward(
    mbart_attention_module,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor = None,
    past_key_value = None,
    attention_mask: torch.Tensor = None,
    layer_head_mask: torch.Tensor = None,
    output_attentions: bool = False,
):
    bsz, tgt_len, _ = hidden_states.size()

    # Prepare Q, K, V projections
    query = mbart_attention_module.q_proj(hidden_states)
    
    if key_value_states is not None and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
        k, v = past_key_value
    elif key_value_states is not None:
        k = mbart_attention_module._shape(
            mbart_attention_module.k_proj(key_value_states), -1, bsz
        )
        v = mbart_attention_module._shape(
            mbart_attention_module.v_proj(key_value_states), -1, bsz
        )
    elif past_key_value is not None:
        k_ = mbart_attention_module._shape(
            mbart_attention_module.k_proj(hidden_states), -1, bsz
        )
        v_ = mbart_attention_module._shape(
            mbart_attention_module.v_proj(hidden_states), -1, bsz
        )
        k = torch.cat([past_key_value[0], k_], dim=2)
        v = torch.cat([past_key_value[1], v_], dim=2)
    else:
        k = mbart_attention_module._shape(
            mbart_attention_module.k_proj(hidden_states), -1, bsz
        )
        v = mbart_attention_module._shape(
            mbart_attention_module.v_proj(hidden_states), -1, bsz
        )

    # Cache keys and values for decoder
    if mbart_attention_module.is_decoder:
        past_key_value = (k, v)

    # Reshape for attention computation
    proj_shape = (bsz * mbart_attention_module.num_heads, -1, mbart_attention_module.head_dim)
    q = mbart_attention_module._shape(query, tgt_len, bsz).view(*proj_shape)
    k = k.view(*proj_shape)
    v = v.view(*proj_shape)
    
    src_len = k.size(1)
    
    # Debug print to check tensor shapes
    print(f"Debug - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

    # Prepare attention bias from mask - FIXED VERSION
    attn_bias = None
    if attention_mask is not None:
        # Ensure attention_mask has correct shape [bsz, 1, tgt_len, src_len]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.view(bsz, 1, 1, src_len).expand(bsz, 1, tgt_len, src_len)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4:
            pass  # Already correct shape
        
        # Create attention bias
        attn_bias = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attn_bias = attn_bias.masked_fill(attention_mask == 1, 0.0)
        
        # Reshape to match the attention computation: [bsz * num_heads, tgt_len, src_len]
        attn_bias = attn_bias.expand(bsz, mbart_attention_module.num_heads, tgt_len, src_len)
        attn_bias = attn_bias.contiguous().view(bsz * mbart_attention_module.num_heads, tgt_len, src_len)

    # Try efficient attention, else fallback
    try:
        if XFORMERS_AVAILABLE:
            # Reshape for xformers (expects B, H, S, D format)
            q_xf = q.view(bsz, mbart_attention_module.num_heads, tgt_len, mbart_attention_module.head_dim)
            k_xf = k.view(bsz, mbart_attention_module.num_heads, src_len, mbart_attention_module.head_dim)
            v_xf = v.view(bsz, mbart_attention_module.num_heads, src_len, mbart_attention_module.head_dim)
            
            # Reshape attention bias for xformers
            xf_attn_bias = None
            if attn_bias is not None:
                xf_attn_bias = attn_bias.view(bsz, mbart_attention_module.num_heads, tgt_len, src_len)
            
            attn_output = xops.memory_efficient_attention(
                q_xf,
                k_xf,
                v_xf,
                p=mbart_attention_module.dropout if mbart_attention_module.training else 0.0,
                attn_bias=xf_attn_bias,
                scale=1.0 / math.sqrt(mbart_attention_module.head_dim),
            )
            # Reshape back to (bsz * num_heads, tgt_len, head_dim)
            attn_output = attn_output.view(bsz * mbart_attention_module.num_heads, tgt_len, mbart_attention_module.head_dim)
        else:
            # Use our custom fallback
            attn_output = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=mbart_attention_module.dropout if mbart_attention_module.training else 0.0,
                scale=1.0 / math.sqrt(mbart_attention_module.head_dim),
                head_dim=mbart_attention_module.head_dim,
                training=mbart_attention_module.training,
            )
        attn_weights = None
    except Exception as e:
        print(f"Efficient attention failed: {e}. Falling back to manual calculation.")
        
        # Manual attention computation
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / math.sqrt(mbart_attention_module.head_dim)

        if attn_bias is not None:
            try:
                if attn_bias.shape == attn_weights.shape:
                    attn_weights = attn_weights + attn_bias
                else:
                    print(f"Skipping attn_bias due to shape mismatch: bias {attn_bias.shape} vs weights {attn_weights.shape}")
            except Exception as exc:
                print(f"Error applying attn_bias in fallback: {exc}")

        attn_weights = F.softmax(attn_weights, dim=-1)
        if mbart_attention_module.training and mbart_attention_module.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=mbart_attention_module.dropout, training=True)

        attn_output = torch.bmm(attn_weights, v)

    # Reshape output back to original format
    attn_output = attn_output.view(bsz, mbart_attention_module.num_heads, tgt_len, mbart_attention_module.head_dim)
    attn_output = attn_output.transpose(1, 2)
    
    # Reshape and project output - CRITICAL FIX HERE
    attn_output = attn_output.reshape(bsz, tgt_len, mbart_attention_module.embed_dim)
    
    print(f"Debug - Final attn_output shape before out_proj: {attn_output.shape}")
    print(f"Debug - out_proj weight shape: {mbart_attention_module.out_proj.weight.shape}")
    
    attn_output = mbart_attention_module.out_proj(attn_output)

    return attn_output, attn_weights, past_key_value

def mbart_decoder_fast_forward(
    mbart_decoder_module: MBartDecoder,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    # ... [Include the complete function as provided in your original code]
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else mbart_decoder_module.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else mbart_decoder_module.config.output_hidden_states
    )
    use_cache = (
        use_cache if use_cache is not None else mbart_decoder_module.config.use_cache
    )
    return_dict = (
        return_dict
        if return_dict is not None
        else mbart_decoder_module.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input = input_ids
        input_shape = input.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    # past_key_values_length
    past_key_values_length = (
        past_key_values[0][0].shape[2] if past_key_values is not None else 0
    )

    if inputs_embeds is None:
        inputs_embeds = (
            mbart_decoder_module.embed_tokens(input_ids)
            * mbart_decoder_module.embed_scale
        )

    # No need to make attention_mask for fast attention -> revived to allow custom attention masking
    attention_mask = mbart_decoder_module._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

    # embed positions
    positions = mbart_decoder_module.embed_positions(input, past_key_values_length)

    hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
    hidden_states = mbart_decoder_module.layernorm_embedding(hidden_states)

    hidden_states = nn.functional.dropout(
        hidden_states,
        p=mbart_decoder_module.dropout,
        training=mbart_decoder_module.training,
    )

    if mbart_decoder_module.gradient_checkpointing and mbart_decoder_module.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = (
        () if (output_attentions and encoder_hidden_states is not None) else None
    )
    next_decoder_cache = () if use_cache else None

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip(
        [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
    ):
        if attn_mask is not None:
            if attn_mask.size()[0] != len(mbart_decoder_module.layers):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(mbart_decoder_module.layers)} layers, but it is for"
                    f" {attn_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(mbart_decoder_module.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        dropout_probability = random.uniform(0, 1)
        if mbart_decoder_module.training and (
            dropout_probability < mbart_decoder_module.layerdrop
        ):
            continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if (
            mbart_decoder_module.gradient_checkpointing
            and mbart_decoder_module.training
        ):

            def create_custom_forward(module: nn.Module) -> Callable:
                def custom_forward(*inputs: Any) -> Any:
                    # None for past_key_value
                    return module(*inputs, output_attentions, use_cache)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                head_mask[idx] if head_mask is not None else None,
                cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

    hidden_states = mbart_decoder_module.layer_norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attentions,
    )
import torch
from model_center.layer import Encoder, Decoder, Embedding, Linear, SegmentPositionEmbedding 
from model_center.layer import LayerNorm
import bmtrain as bmt

class CPMLive(torch.nn.Module):
    
    def __init__(self, config):
        
        super().__init__()

        self.encoder = Encoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
            mask_modules = config.mask_modules,)

        self.prompt_embedding = Embedding(
            vocab_size = config.prompt_types * config.prompt_length, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,)

        self.segment_embedding = Embedding(
            vocab_size = config.segment_types, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,)

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,)

        self.position_bias = SegmentPositionEmbedding(
            num_segments = config.segment_types,
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            absolute_inner_segment = config.absolute_inner_segment,
            bidirectional = True,
            dtype = config.dtype,)
        
        self.prompt_length = config.prompt_length
        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.output_projection = Linear(
                vocab_size = config.cls_head,
                embedding_size = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,)
        elif not self.tied:
            self.output_projection = Linear(
                vocab_size = config.vocab_size,
                embedding_size = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,)

    def forward(self, input : torch.Tensor, # (batch, seqlen)
                      length : torch.Tensor, # (batch)
                      context : torch.Tensor, # (batch, seqlen)
                      position: torch.Tensor, # (batch, seqlen)
                      segment: torch.Tensor, # (batch, seqlen)
                      span : torch.Tensor,  # (batch, seqlen)
                ):

        batch = input.size(0)
        seqlen = input.size(1)
        input_prompt = input[:, :self.prompt_length].contiguous()
        input_ids = input[:, self.prompt_length:].contiguous()

        prompt_states = self.prompt_embedding(input_prompt)
        hidden_states = self.input_embedding(input_ids)
        segment_states = self.segment_embedding(segment)
        
        hidden_states = torch.cat([prompt_states, hidden_states], 1) + segment_states

        with torch.no_grad():
            device = input.device
            directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
            attention_mask = context[:, None, :] | (context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen))
            attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
            mask_1d = torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
            attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask

        position_bias = self.position_bias(position, position, segment, segment)
        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)

        if self.cls_head:
            logits = self.output_projection(hidden_states)
        elif not self.tied:
            logits = self.output_projection(hidden_states)
        else:
            logits = self.input_embedding.projection(hidden_states)

        return logits, hidden_states

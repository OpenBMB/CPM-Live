# coding=utf-8
# Copyright 2022 The OpenBMB team.
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

from typing import List, Optional, Tuple
import torch
from ..utils import Config
from ..layers import Encoder, Embedding, BucketPositionBias
import bmtrain as bmt


class CPMBeeConfig(Config):
    def __init__(
        self,
        vocab_size=30720,
        dim_model=4096,
        num_heads=64,
        dim_head=64,
        dim_ff=10240,
        num_layers=32,
        dropout_p=0.0,
        position_bias_num_buckets=512,
        position_bias_num_segment_buckets=64,
        position_bias_max_distance=2048,
        eps=1e-6,
        half: bool = True,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_num_segment_buckets = position_bias_num_segment_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        if half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules


class CPMBee(bmt.DistributedModule):
    def __init__(self, config: CPMBeeConfig):

        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            mask_modules=config.mask_modules,
        )

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            init_std=0.02,
        )

        self.position_bias = BucketPositionBias(
            num_heads=config.num_heads,
            num_buckets=config.position_bias_num_buckets,
            num_segment_bucket=config.position_bias_num_segment_buckets,
            max_distance=config.position_bias_max_distance,
            dtype=config.dtype,
        )

    def forward(
        self,
        input: torch.Tensor,  # (batch, seqlen) long
        length: torch.Tensor,  # (batch) long
        context: torch.Tensor,  # (batch, seqlen) bool
        sample_idx : torch.Tensor,  # (batch, seq_len) long
        num_segments : torch.Tensor,    # (batch, seq_len) long  number of segments in batch
        segment: torch.Tensor,  # (batch, seqlen) long  segment id of each token
        segment_rel_offset : torch.Tensor,  # (batch, seq_len) long segment rel offset in this instance
        segment_rel : torch.Tensor, # (batch, num_segment_bucket) long  mapping n_segment x n_segment to bucket id
        span: torch.Tensor,  # (batch, seqlen) long span id of each instance
    ):

        batch = input.size(0)
        seqlen = input.size(1)

        with torch.no_grad():
            device = input.device

            # calc segment bucket
            segment_rel_2d = torch.masked_fill(
                segment[:, :, None] * num_segments[:, :, None] + segment[:, None, :] + segment_rel_offset[:, :, None],
                ~((sample_idx[:, :, None] == sample_idx[:, None, :]) & (span[:, None, :] == span[:, :, None])), # not in the same span and sample
                0,  # avoid torch.gather overflow
            ).view(batch, seqlen * seqlen)

            segment_bucket = torch.gather(
                input = segment_rel,
                dim = 1,
                index = segment_rel_2d,
            ).view(batch, seqlen, seqlen)
            
            segment_bucket.masked_fill_(
                ~((sample_idx[:, :, None] == sample_idx[:, None, :]) & (span[:, None, :] == span[:, :, None])), # not in the same span and sample
                1,  # bucket is used for in-context samples
            )

            # directional mask
            directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(
                seqlen, device=device
            ).view(-1, 1)
            # sample mask
            sample_mask_2d = (sample_idx[:, :, None] == 0) | (sample_idx[:, :, None] == sample_idx[:, None, :])
            # context mask
            attention_mask = context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
            )
            # span mask
            attention_mask = attention_mask & sample_mask_2d & (span[:, None, :] == span[:, :, None])
            # length mask
            mask_1d = (
                torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
            )
            attention_mask = (
                mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
            )
            position = torch.arange(seqlen, device=device).expand(batch, seqlen)

        hidden_states = self.input_embedding(input)
        position_bias = self.position_bias(position, position, segment_bucket)

        hidden_states = self.encoder(
            hidden_states, attention_mask, position_bias
        )
        logits = self.input_embedding.projection(hidden_states)
        return logits, hidden_states

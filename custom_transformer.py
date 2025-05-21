import math
import torch
import torch.nn as nn
from custom_layers import FMoE
from custom_layers import FMoELinear

class _Expert(nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size,
        inner_hidden_size,
        activation,
        rank = 0,
    ):
        super().__init__()
        self.expand = FMoELinear(
            num_experts,
            hidden_size,
            inner_hidden_size, 
            bias = True,
            rank = rank,
        )
        self.shrink = FMoELinear(
            num_experts,
            inner_hidden_size,
            hidden_size,
            bias = True,
            rank = rank,
        )
        self.activation = activation
    
    def forward(self, inp, fwd_expert_count):
        out = self.expand(inp, fwd_expert_count)
        out = self.activation(out)
        out = self.shrink(out, fwd_expert_count)
        return out

class FMoETransformerMLP(FMoE):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        activation,
        gate,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        world_size,
        expert_dp_comm = "none",
        expert_rank = 0,
    ):
        super().__init__(
            num_expert = num_experts,
            d_model = hidden_size // mhmoe_num_heads,
            moe_top_k = moe_top_k,
            gate = gate,
            world_size=world_size,
        )

        self.experts = _Expert(
            hidden_size = hidden_size // mhmoe_num_heads,
            inner_hidden_size = int(inner_hidden_size * mhmoe_beta),
            activation = activation,
            num_experts = num_experts,
            rank = expert_rank,
        )

        self.hidden_size = hidden_size
        self.mhmoe_num_heads = mhmoe_num_heads
        
        self.inner_head_dim = inner_hidden_size // mhmoe_num_heads
        if self.mhmoe_num_heads > 1:
            self.split_layer = nn.Linear(hidden_size, hidden_size)
            nn.init.xavier_uniform_(self.split_layer.weight, gain = 1 / math.sqrt(2))
            self.merge_layer = nn.Linear(hidden_size, hidden_size)
            nn.init.xavier_uniform_(self.merge_layer.weight)
            nn.init.constant_(self.merge_layer.bias, 0.0)
        
        self.mark_parallel_comm(expert_dp_comm)
    
    def forward(self, inp): 
        original_shape = inp.shape
        reshaped_inp = inp.reshape(-1, self.hidden_size)
        if self.mhmoe_num_heads > 1:
            inp = self.split_layer(reshaped_inp)
            N, dim = reshaped_inp.shape

            inp = inp.reshape(N, self.mhmoe_num_heads, dim // self.mhmoe_num_heads).contiguous()
            inp = inp.reshape(N * self.mhmoe_num_heads, dim // self.mhmoe_num_heads).contiguous()
            
            out = super().forward(inp)

            out = out.reshape(N, self.mhmoe_num_heads, dim // self.mhmoe_num_heads).contiguous()
            out = out.reshape(N, self.hidden_size).contiguous()
            out = self.merge_layer(out)
            out = out.reshape(original_shape)
        else:
            out = super().forward(reshaped_inp)
            out = out.reshape(original_shape)
        return out
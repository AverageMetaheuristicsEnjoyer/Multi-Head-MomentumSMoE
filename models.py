import torch
import torch.nn as nn
import torch.nn.functional as F

from gates import TopKGate
from experts import FeedForwardExpert

# TODO: add commentaries to lines where permuting dimensions of a tensor
# e.g. in reshape: [dim_1, dim_2, dim_3] --> [dim_1, dim_2 / dim_3]

# TODO?: add references to the formulas for better navigation

class MomentumSMoELayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_experts,
        k = 2,
        dropout_rate = 0.1,
        mu = 0.7,
        gamma = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.mu = mu
        self.gamma = gamma
        
        self.experts = nn.ModuleList([
            FeedForwardExpert(
                input_dim,
                hidden_dim,
                output_dim,
                dropout_rate,
            )
            for _ in range(num_experts)
        ])
        
        self.gate = TopKGate(input_dim, num_experts, k)
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, momentum = None):
        if momentum is None:
            momentum = torch.zeros_like(x)
            
        gates, _ = self.gate(x)
        
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs += gates[..., i:i + 1] * expert_out
        
        momentum = -expert_outputs + self.mu * momentum
        output = x + self.gamma * momentum
        
        output = self.layer_norm(output)
        
        return output, momentum

class MultiHeadSplitLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.multi_head_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x_hat = self.multi_head_proj(x)
        
        x_split = x_hat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        x_flat = x_split.reshape(batch_size * seq_len * self.num_heads, self.head_dim)
        
        return x_flat

class MultiHeadMergeLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.merge_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, o, batch_size, seq_len):
        o_reshaped = o.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        o_concat = o_reshaped.reshape(batch_size, seq_len, self.input_dim)
        
        o_merged = self.merge_proj(o_concat)
        
        return o_merged

class MultiHeadMomentumSMoELayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim, 
        num_experts, 
        num_heads,
        k = 2, 
        dropout_rate = 0.1, 
        mu = 0.7, 
        gamma = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.mu = mu
        self.gamma = gamma
        
        self.split_layer = MultiHeadSplitLayer(input_dim, num_heads)
        
        self.moe_layer = MomentumSMoELayer(
            self.head_dim, 
            hidden_dim // num_heads, 
            self.head_dim,
            num_experts, 
            k, 
            dropout_rate, 
            mu, 
            gamma
        )
        
        self.merge_layer = MultiHeadMergeLayer(input_dim, num_heads)
        
    def forward(self, x, momentum=None):
        batch_size, seq_len, _ = x.shape
        
        sub_tokens = self.split_layer(x)
        
        if momentum is None:
            momentum = torch.zeros_like(sub_tokens)

        moe_output, momentum = self.moe_layer(sub_tokens, momentum)
        
        final_output = self.merge_layer(moe_output, batch_size, seq_len)
        
        return final_output, momentum

class MHMomentumSMoE(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_classes,
        num_experts, 
        num_heads, 
        num_layers, 
        k = 2,
        dropout_rate = 0.1, 
        mu = 0.7,
        gamma = 1.0
    ):
        super().__init__()
        
        self.input_embed = nn.Linear(input_size, hidden_size)
        
        self.layers = nn.ModuleList([
            MultiHeadMomentumSMoELayer(
                hidden_size, hidden_size * 4, hidden_size, 
                num_experts, num_heads, k, dropout_rate, mu, gamma
            )
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)
        
        x = self.input_embed(x)
        
        x = x.unsqueeze(1)  # [B, 1, H]
        
        momentum_list = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, momentum_list[i] = layer(x, momentum_list[i])
        
        x = x.squeeze(1)
        
        logits = self.classifier(x)
        
        return logits
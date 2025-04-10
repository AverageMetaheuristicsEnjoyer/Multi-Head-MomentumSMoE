import torch
import torch.nn as nn
import torch.nn.functional as F

from gates import TopKGate
from experts import FeedForwardExpert

# TODO: add commentaries to lines where permuting dimensions of a tensor
# e.g. in reshape: [dim_1, dim_2, dim_3] --> [dim_1, dim_2 / dim_3]

# TODO?: add references to the formulas for better navigation

class MomentumLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_experts,
        moe_top_k = 2,
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

        self.gate = TopKGate(input_dim, num_experts, moe_top_k)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, momentum=None):
        gates, _ = self.gate(x)

        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs += gates[..., i:i+1] * expert_out

        momentum = self.mu * momentum + self.gamma * expert_outputs
        output = x - momentum

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

        x_reshaped = self.multi_head_proj(x)
        x_reshaped = x_reshaped.reshape(batch_size * seq_len * self.num_heads, self.head_dim)

        return x_reshaped

class MultiHeadMergeLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.merge_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, batch_size, seq_len):
        x_merged = x.reshape(batch_size, seq_len, self.input_dim)
        x_merged = self.merge_proj(x_merged)

        return x_merged

class MultiHeadMomentumLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_experts,
        num_heads,
        moe_top_k = 2,
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
        self.moe_layer = MomentumLayer(
            self.head_dim,
            hidden_dim // num_heads,
            self.head_dim,
            num_experts,
            moe_top_k,
            dropout_rate,
            mu,
            gamma
        )
        self.merge_layer = MultiHeadMergeLayer(input_dim, num_heads)

    def forward(self, x, momentum = None):
        batch_size, seq_len, _ = x.shape

        sub_tokens = self.split_layer(x)

        if momentum is None:
            momentum = torch.zeros_like(sub_tokens)

        moe_output, momentum = self.moe_layer(sub_tokens, momentum)
        final_output = self.merge_layer(moe_output, batch_size, seq_len)

        return final_output, momentum

# "Replacement" of the multi-head attention block in a layer (multi-head attention + MoE)
# since experiments're on the early stage
class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        identity = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x + identity)
        return x

class MHMomentumSMoE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_experts,
        num_heads,
        num_layers,
        moe_top_k = 2,
        dropout = 0.1,
        mu = 0.7,
        gamma = 1.0
    ):
        super().__init__()

        # just for not sending num_channels through the argument
        self.num_channels = 1 if input_size <= 28*28*3 else 3
        self.img_size = int((input_size / self.num_channels) ** 0.5)

        self.inp_embed = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.feature_map_size = (self.img_size // 4) ** 2
        self.conv_output_size = self.feature_map_size * 32

        self.proj = nn.Linear(self.conv_output_size, hidden_size)
        
        self.ffn_layers = nn.ModuleList()
        self.moe_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.ffn_layers.append(
                FeedForward(hidden_size, hidden_size * 4, hidden_size, dropout)
            )
            
            self.moe_layers.append(
                MultiHeadMomentumLayer(
                    hidden_size, hidden_size * 4, hidden_size,
                    num_experts, num_heads, moe_top_k, dropout, mu, gamma
                )
            )

        self.out_embed = nn.Linear(hidden_size, num_classes)
        
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.inp_embed(x)
        
        x = x.view(batch_size, -1)
        x = self.proj(x)
        
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        momentum_list = [None] * self.num_layers
        
        for i in range(self.num_layers):
            x = self.ffn_layers[i](x)
            
            x, momentum_list[i] = self.moe_layers[i](x, momentum_list[i])

        x = x.squeeze(1)
        out = self.out_embed(x)
        
        return out
import math
import torch
import torch.nn as nn
from custom_layers import FMoE, GroupsFMoE
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
            bias = False,
            rank = rank,
        )
        self.shrink = FMoELinear(
            num_experts,
            inner_hidden_size,
            hidden_size,
            bias = False,
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
        use_xmoe,
        xmoe_dim,
        world_size,
        expert_dp_comm = "none",
        expert_rank = 0,
        **kwargs
    ):
        super().__init__(
            num_expert = num_experts,
            d_model = hidden_size // mhmoe_num_heads,
            moe_top_k = moe_top_k,
            gate = gate,
            world_size=world_size,
            use_xmoe = use_xmoe,
            xmoe_dim = xmoe_dim,
            **kwargs
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
            reshaped_inp = self.split_layer(reshaped_inp)
            N, dim = reshaped_inp.shape

            reshaped_inp = reshaped_inp.reshape(N, self.mhmoe_num_heads, dim // self.mhmoe_num_heads).contiguous()
            reshaped_inp = reshaped_inp.reshape(N * self.mhmoe_num_heads, dim // self.mhmoe_num_heads).contiguous()
            
            out = super().forward(reshaped_inp)

            out = out.reshape(N, self.mhmoe_num_heads, dim // self.mhmoe_num_heads).contiguous()
            out = out.reshape(N, self.hidden_size).contiguous()
            out = self.merge_layer(out)
        else:
            out = super().forward(reshaped_inp)
        out = out.reshape(original_shape)
        return out

class NoMomentum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, moe_out, inner_hist, batch_size, seq_len):
        return moe_out, inner_hist

class HBInner(nn.Module):
    def __init__(
        self,
        gamma,
        mu,
        **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
    
    def forward(self, moe_out, inner_hist, batch_size, seq_len):
        inner_hist = -moe_out + self.mu * inner_hist
        out = self.gamma * inner_hist
        return out, inner_hist

class MarsInner(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_groups,
        gamma1,
        gamma2,
        beta1,
        beta2,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.beta1 = beta1
        self.beta2 = beta2
    
    def forward(self, moe_out, inner_hist, batch_size, seq_len):
        m, v, moe_out_prev = inner_hist
        outps_diff = -moe_out - (-moe_out_prev)
        c = -moe_out + self.gamma2 * (self.beta1 / (1 - self.beta1)) * outps_diff
        
        c_4d = c.reshape(batch_size, seq_len, self.num_groups, self.hidden_size)
        
        # Permute to [G, B, M, H] to isolate groups for batched norm
        c_permuted = c_4d.permute(2, 0, 1, 3)
        
        # Compute matrix norm on the [M, H] slice for each group and sequence
        c_norm = torch.linalg.matrix_norm(c_permuted, dim=(-2, -1), ord="fro")

        batch_idx = c_norm > 1
        scaling_facs = torch.ones_like(c_norm)
        scaling_facs[batch_idx] = c_norm[batch_idx]
        c_t_permuted = c_permuted / scaling_facs.unsqueeze(-1).unsqueeze(-1)

        c_t_4d = c_t_permuted.permute(1, 2, 0, 3)
        c_t = c_t_4d.reshape(-1, self.num_groups, self.hidden_size)

        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2
        out = self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))
        
        inner_hist = (m_t, v_t, moe_out.detach())
        return out, inner_hist

class InnerGroupLayer(GroupsFMoE):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        world_size,
        expert_dp_comm = "none",
        expert_rank = 0,
        inner_mom=None,
        inner_gamma1=1.0,
        inner_gamma2=1.0,
        inner_mu=0.9,
        inner_beta1=0.9,
        inner_beta2=0.999,
        **kwargs,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert = num_experts,
            d_model = hidden_size,
            moe_top_k = moe_top_k,
            gate = gate,
            world_size=world_size,
            **kwargs
        )

        self.experts = _Expert(
            hidden_size = hidden_size,
            inner_hidden_size = inner_hidden_size,
            activation = activation,
            num_experts = num_experts,
            rank = expert_rank,
        )

        self.hidden_size = hidden_size
        self.num_groups = world_size
        self.inner_gamma1 = inner_gamma1
        self.inner_gamma2 = inner_gamma2
        self.inner_mu = inner_mu
        self.inner_beta1 = inner_beta1
        self.inner_beta2 = inner_beta2

        if inner_mom == "heavy-ball":
            self.inner_mom_cls = HBInner(self.inner_gamma1, self.inner_mu)
        elif inner_mom == "mars":
            self.inner_mom_cls = MarsInner(
                self.hidden_size,
                self.num_groups,
                inner_gamma1,
                inner_gamma2,
                inner_beta1,
                inner_beta2,
            )
        else:
            self.inner_mom_cls = NoMomentum()


        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp, inner_hist, batch_size, seq_len):        
        expert_outputs, gate_scores = super().forward(inp)
        weighted_group_outputs = torch.sum(expert_outputs * gate_scores, dim=2)

        group_moe_out, inner_hist = self.inner_mom_cls(
            weighted_group_outputs,
            inner_hist,
            batch_size,
            seq_len,
        )
        
        return group_moe_out, inner_hist
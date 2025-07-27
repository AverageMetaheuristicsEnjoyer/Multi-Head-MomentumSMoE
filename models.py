import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_transformer import FMoETransformerMLP, InnerGroupLayer
from gates import *

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span

def _skew(X, pad_value):
    """shift every row one step to the right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value = pad_value) # B x M x (M + L + 1)
    X = X.view(B, -1) # B x (ML + MM + M)
    X = X[:, :-M] # B x (ML + MM)
    X = X.view(B, M, M + L) # B x M x (L + M)
    return X

def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x (M + L)
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x (ML + MM)
    X = F.pad(X, (0, M))  # B x (ML + MM + M)
    X = X.view(B, M, M + L + 1)  # B x M x (L + M + 1)
    X = X[:, :, :L]  # B x M x L
    return X

class SeqAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        attn_span,
        dropout
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_span = attn_span
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, pos_encoding):
        # query size = B x M x H
        # key, value sizes = B x (M + L) x H

        # B x M (q) x (M + L) (k)
        attn_ctx = torch.matmul(query, key.transpose(-1, -2))
        attn_ctx = _unskew(attn_ctx) # B x M x L

        attn_pos = torch.matmul(query, pos_encoding) # B x M x L_pos
        attn = attn_ctx + attn_pos

        attn = attn / math.sqrt(self.hidden_size) # B x M x L_pos
        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)

        attn_ctx = _skew(attn, 0) # B x M x (L + M)
        out = torch.matmul(attn_ctx, value) # B x M x H
        return out

    def get_cache_size(self):
        return self.attn_span

class MultiHeadSeqAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout,
        attn_span,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn = SeqAttention(
            hidden_size = self.head_dim,
            dropout = dropout,
            attn_span = attn_span,
        )
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias = False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias = False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias = False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias = False)

    def head_reshape(self, x):
        K = self.num_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D)) # B x (M + L) x K x D
        x = x.transpose(1, 2).contiguous() # B x K x (M + L) x D
        x = x.view(-1, x.size(-2), x.size(-1)) # B_K x (M + L) x D
        return x

    def forward(self, query, key, value, pos_encoding):
        B = query.size(0)
        K = self.num_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, pos_encoding) # B_K x M x D
        out = out.view(B, K, M, D) # B x K x M x D
        out = out.transpose(1, 2).contiguous() # B x M x K x D
        out = out.view(B, M, -1) # B x M x K_D
        out = self.proj_out(out)
        return out

class NoMomentum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, moe_out, outer_hist, original_shape = None):
        return moe_out, outer_hist

class HBOuter(nn.Module):
    def __init__(
        self,
        gamma,
        mu,
        **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
    
    def forward(self, moe_out, outer_hist, original_shape):
        outer_hist = -moe_out + self.mu * outer_hist
        out = self.gamma * outer_hist
        return out, outer_hist

class MarsOuter(nn.Module):
    def __init__(
        self,
        hidden_size,
        gamma1,
        gamma2,
        beta1,
        beta2,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.beta1 = beta1
        self.beta2 = beta2
    
    def forward(self, moe_out, outer_hist, original_shape):
        m, v, moe_out_prev = outer_hist
        moe_out_prev = moe_out_prev.reshape(-1, self.hidden_size)

        outps_diff = -moe_out - (-moe_out_prev)

        c = -moe_out + self.gamma2 * (self.beta1 / (1 - self.beta1)) * outps_diff
        c_norm = torch.linalg.matrix_norm(c, dim = (-2, -1), ord = "fro")
        batch_idx = c_norm > 1
        scaling_facs = torch.ones_like(c_norm)
        scaling_facs[batch_idx] = c_norm[batch_idx]
        c_t = c / scaling_facs.view(-1, 1, 1)
        c_t = c_t.reshape(-1, self.hidden_size)
        
        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2
        out = self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))
        outer_hist = (m_t, v_t, moe_out.detach().reshape(original_shape))
        return out, outer_hist

class OuterLayer(InnerGroupLayer):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        world_size,
        inner_mom=None,
        inner_gamma1=1.0,
        inner_gamma2=1.0,
        inner_mu=0.9,
        inner_beta1=0.9,
        inner_beta2=0.999,
        outer_mom=None,
        outer_gamma1=1.0,
        outer_gamma2=1.0,
        outer_mu=0.9,
        outer_beta1=0.9,
        outer_beta2=0.999,
        **kwargs
    ):
        super().__init__(
            hidden_size=hidden_size,
            inner_hidden_size=inner_hidden_size,
            dropout=dropout,
            gate=gate,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
            world_size=world_size,
            inner_mom=inner_mom,
            inner_gamma1=inner_gamma1, inner_gamma2=inner_gamma2, inner_mu=inner_mu,
            inner_beta1=inner_beta1, inner_beta2=inner_beta2,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.outer_mom = outer_mom
        self.outer_gamma1 = outer_gamma1
        self.outer_gamma2 = outer_gamma2
        self.outer_mu = outer_mu
        self.outer_beta1 = outer_beta1
        self.outer_beta2 = outer_beta2

        if self.outer_mom == "heavy-ball":
            self.outer_mom_cls = HBOuter(self.outer_gamma1, self.outer_mu)
        elif self.outer_mom == "mars":
            self.outer_mom_cls = MarsOuter(
                hidden_size,
                self.outer_gamma1,
                self.outer_gamma2,
                self.outer_beta1,
                self.outer_beta2,
            )
        else:
            self.outer_mom_cls = NoMomentum()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inp, hist):
        original_shape = inp.shape
        batch_size = original_shape[0]
        seq_len = original_shape[1]

        reshaped_inp = inp.reshape(-1, self.hidden_size)
        outer_hist, inner_hist = hist

        groups_out, inner_hist = super().forward(reshaped_inp, inner_hist, batch_size, seq_len)
        moe_out = torch.sum(groups_out, dim=1)
        moe_out = self.dropout(moe_out)

        out, outer_hist = self.outer_mom_cls(moe_out, outer_hist, original_shape)
        out = out.reshape(original_shape)

        return out, (outer_hist, inner_hist)

class MomentumLayer(FMoETransformerMLP):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma,
        mu,
        use_xmoe,
        xmoe_dim,
        world_size,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(   
            hidden_size = hidden_size,
            inner_hidden_size = inner_hidden_size,
            activation = activation,
            gate = gate,
            num_experts = num_experts,
            moe_top_k = moe_top_k,
            mhmoe_num_heads = mhmoe_num_heads,
            mhmoe_beta = mhmoe_beta,
            use_xmoe = use_xmoe,
            xmoe_dim = xmoe_dim,
            world_size = world_size,
        )
        self.gamma = gamma
        self.mu = mu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, momentum):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)

        momentum = -moe_out + self.mu * momentum
        output = self.gamma * momentum
        return output, momentum

class MarsLayer(FMoETransformerMLP):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        use_xmoe,
        xmoe_dim,
        world_size,
        beta1,
        beta2,
        layerth,
        **kwargs
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            hidden_size = hidden_size,
            inner_hidden_size = inner_hidden_size,
            activation = activation,
            gate = gate,
            num_experts = num_experts,
            moe_top_k = moe_top_k,
            mhmoe_num_heads = mhmoe_num_heads,
            mhmoe_beta = mhmoe_beta,
            use_xmoe = use_xmoe,
            xmoe_dim = xmoe_dim,
            world_size = world_size,
            **kwargs
        )
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.layerth = layerth
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hist):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)
        
        m, v, _, moe_out_prev = hist

        outps_diff = -moe_out - (-moe_out_prev)

        c = -moe_out + self.gamma2 * (self.beta1 / (1 - self.beta1)) * outps_diff
        c_norm = torch.linalg.matrix_norm(c, dim = (-2, -1), ord = "fro")
        batch_idx = c_norm > 1
        scaling_facs = torch.ones_like(c_norm)
        scaling_facs[batch_idx] = c_norm[batch_idx]
        c_t = c / scaling_facs.view(-1, 1, 1)
        
        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2

        out = self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))

        return out, (m_t, v_t, _, moe_out.detach())

class OptimistAdamLayer(FMoETransformerMLP):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        use_xmoe,
        xmoe_dim,
        world_size,
        beta1,
        beta2,
        layerth,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            hidden_size = hidden_size,
            inner_hidden_size = inner_hidden_size,
            activation = activation,
            gate = gate,
            num_experts = num_experts,
            moe_top_k = moe_top_k,
            mhmoe_num_heads = mhmoe_num_heads,
            mhmoe_beta = mhmoe_beta,
            use_xmoe = use_xmoe,
            xmoe_dim = xmoe_dim,
            world_size = world_size,
        )
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.layerth = layerth
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layerth = layerth

    def forward(self, inp, hist):        
        moe_out = super().forward(self.layer_norm(inp))
        moe_out = self.dropout(moe_out)
        
        m, v, momentum, moe_out_prev = hist
        
        out = (1 + self.gamma2) * (-moe_out) - self.gamma2 * (-moe_out_prev)
        m_t = m
        v_t = v

        return out, (m_t, v_t, momentum, moe_out.detach())

class OptimistAdamLayerV2(FMoETransformerMLP):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        gate,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        use_xmoe,
        xmoe_dim,
        world_size,
        beta1,
        beta2,
        layerth,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            hidden_size = hidden_size,
            inner_hidden_size = inner_hidden_size,
            activation = activation,
            gate = gate,
            num_experts = num_experts,
            moe_top_k = moe_top_k,
            mhmoe_num_heads = mhmoe_num_heads,
            mhmoe_beta = mhmoe_beta,
            use_xmoe = use_xmoe,
            xmoe_dim = xmoe_dim,
            world_size = world_size,
        )
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.layerth = layerth
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.alpha = 0.5

    def forward(self, inp, hist):
        moe_out = super().forward(self.layer_norm(inp))
        moe_out = self.dropout(moe_out)
        
        m, v, mom, moe_out_prev = hist

        c_t = (1 + self.alpha) * (-moe_out) - self.alpha * (-moe_out_prev)
        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2

        out = self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))

        return out, (m_t, v_t, mom, moe_out.detach())

class TransformerSeqLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        num_heads,
        attn_span,
        dropout,
        gate_name,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        alpha,
        beta1,
        beta2,
        beta3,
        t_warmup,
        alpha_warmup,
        beta3_warmup,
        ademamix_all_layers,
        weight_decay,
        rand_zero,
        use_xmoe,
        xmoe_dim,
        world_size,
        s,
        g,
        layerth,
        inner_mom=None, inner_gamma1=1.0, inner_gamma2=1.0, inner_mu=0.9,
        inner_beta1=0.9, inner_beta2=0.999,
        outer_mom=None, outer_gamma1=1.0, outer_gamma2=1.0, outer_mu=0.9,
        outer_beta1=0.9, outer_beta2=0.999,
        **kwargs
    ):
        super().__init__()
        if gate_name == "smoe":
            gate = CustomNaiveGate_Balance_SMoE # from SwitchTransformer paper
        elif gate_name == "mhmoe":
            gate = MHMoEGate
        elif gate_name == "smome":
            gate = SMoE_Momentum
        elif gate_name == "srome":
            gate = SMoE_Reg
        elif gate_name == "groups":
            gate = GroupsGate
        else:
            ValueError("Incorrect gate name")
        
        self.use_attn = s == "s"
        self.attn = (
            MultiHeadSeqAttention(
                hidden_size = hidden_size,
                num_heads = num_heads,
                dropout = dropout,
                attn_span = attn_span,
            )
            if self.use_attn
            else None
        )
        
        self.use_smoe = g in ["m", "a", "e", "r", "l", "o", "v", "p"]
        self.smoe = (
            MomentumLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma = gamma2,
                mu = mu,
                use_xmoe = use_xmoe,
                xmoe_dim = xmoe_dim,
                world_size = world_size,
            )
            if g == "m"
            else
            MarsLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma1 = gamma1,
                gamma2 = gamma2,
                mu = mu,
                use_xmoe = use_xmoe,
                xmoe_dim = xmoe_dim,
                world_size = world_size,
                beta1 = beta1,
                beta2 = beta2,
                layerth = layerth,
                **kwargs
            )
            if g == "r"
            else
            OptimistAdamLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma1 = gamma1,
                gamma2 = gamma2,
                mu = mu,
                use_xmoe = use_xmoe,
                xmoe_dim = xmoe_dim,
                world_size = world_size,
                beta1 = beta1,
                beta2 = beta2,
                layerth = layerth,
            )
            if g == "o"
            else
            OptimistAdamLayerV2(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma1 = gamma1,
                gamma2 = gamma2,
                mu = mu,
                use_xmoe = use_xmoe,
                xmoe_dim = xmoe_dim,
                world_size = world_size,
                beta1 = beta1,
                beta2 = beta2,
                layerth = layerth,
            )
            if g == "v"
            else
            OuterLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                world_size = world_size,
                inner_mom=inner_mom,
                inner_gamma1=inner_gamma1, inner_gamma2=inner_gamma2, inner_mu=inner_mu,
                inner_beta1=inner_beta1, inner_beta2=inner_beta2,
                outer_mom=outer_mom,
                outer_gamma1=outer_gamma1, outer_gamma2=outer_gamma2, outer_mu=outer_mu,
                outer_beta1=outer_beta1, outer_beta2=outer_beta2,
                **kwargs
            )
            if g == "p"
            else None
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, h, h_cache, pos_encoding, momentum):
        # h = B x M x H
        # h_cache = B x L x H
        if self.use_attn:
            h_all = torch.cat([h_cache, h], dim = 1) # B x (M + L) x H
            attn_out = self.attn(h, h_all, h_all, pos_encoding)
            h = self.norm1(h + attn_out) # B x M x H
        if self.use_smoe:
            smoe_out, momentum = self.smoe(h, momentum)
            h = self.norm2(h + smoe_out) # B x M x H
        return h, momentum

class TransformerSeq(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_hidden_size,
        num_heads,
        num_layers,
        attn_span,
        architecture,
        dropout,
        gate_name,
        num_experts,
        moe_top_k,
        mhmoe_num_heads,
        mhmoe_beta,
        gamma1,
        gamma2,
        mu,
        alpha,
        beta1,
        beta2,
        beta3,
        t_warmup,
        alpha_warmup,
        beta3_warmup,
        ademamix_all_layers,
        weight_decay,
        rand_zero,
        use_xmoe,
        xmoe_dim,
        world_size,
        inner_mom=None, inner_gamma1=1.0, inner_gamma2=1.0, inner_mu=0.9,
        inner_beta1=0.9, inner_beta2=0.999,
        outer_mom=None, outer_gamma1=1.0, outer_gamma2=1.0, outer_mu=0.9,
        outer_beta1=0.9, outer_beta2=0.999,
        **kwargs,
    ):
        super().__init__()
        self.inp_embed = nn.Embedding(vocab_size, hidden_size)
        self.out_embed = nn.Linear(hidden_size, vocab_size)

        self.pos_encoding = nn.Parameter(torch.randn(1, hidden_size // num_heads, attn_span))
        self.arch = architecture

        self.attn_layer_count = self.arch.count("s")
        self.layers = nn.ModuleList()

        self.layers.extend(
            TransformerSeqLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                num_heads = num_heads,
                attn_span = attn_span,
                dropout = dropout,
                gate_name = gate_name,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                gamma1 = gamma1,
                gamma2 = gamma2,
                mu = mu,
                alpha = alpha,
                beta1 = beta1,
                beta2 = beta2,
                beta3 = beta3,
                t_warmup = t_warmup,
                alpha_warmup = alpha_warmup,
                beta3_warmup = beta3_warmup,
                ademamix_all_layers = ademamix_all_layers,
                weight_decay = weight_decay,
                use_xmoe = use_xmoe,
                xmoe_dim = xmoe_dim,
                rand_zero = rand_zero,
                world_size = world_size,
                s = self.arch[2 * i],
                g = self.arch[2 * i + 1],
                layerth = i,
                inner_mom=inner_mom,
                inner_gamma1=inner_gamma1, inner_gamma2=inner_gamma2, inner_mu=inner_mu,
                inner_beta1=inner_beta1, inner_beta2=inner_beta2,
                outer_mom=outer_mom,
                outer_gamma1=outer_gamma1, outer_gamma2=outer_gamma2, outer_mu=outer_mu,
                outer_beta1=outer_beta1, outer_beta2=outer_beta2,
                **kwargs
            )
            for i in range(num_layers)
        )
        self.world_size = world_size
        self.inner_mom = inner_mom
        self.outer_mom = outer_mom
    
    def forward(self, x, h_cache):
        block_size = x.size(1) # B x M
        h = self.inp_embed(x) # B x M x H
        h_cache_next = []
        if "p" in self.arch:
            # --- DYNAMIC STATE INITIALIZATION ---
            
            # 1. Initialize Outer Layer State
            if self.outer_mom == 'heavy-ball':
                # State is 3D: [B, M, H]
                outer_state = torch.zeros_like(h)
            elif self.outer_mom == 'mars':
                # m, v are 2D for token-level updates; prev_out is 3D
                m_outer = torch.zeros_like(h.reshape(-1, h.shape[-1]))
                v_outer = torch.zeros_like(h.reshape(-1, h.shape[-1]))
                prev_out_outer = torch.zeros_like(h)
                outer_state = (m_outer, v_outer, prev_out_outer)
            else:
                outer_state = None

            # 2. Initialize Inner Layer State
            inner_shape_3d = (h.shape[0] * h.shape[1], self.world_size, h.shape[2])
            if self.inner_mom == 'heavy-ball':
                # State is 3D: [Tokens, Groups, H]
                inner_state = torch.zeros(inner_shape_3d, device=h.device)
            elif self.inner_mom == 'mars':
                # All history tensors are 3D: [Tokens, Groups, H]
                m_inner = torch.zeros(inner_shape_3d, device=h.device)
                v_inner = torch.zeros(inner_shape_3d, device=h.device)
                prev_out_inner = torch.zeros(inner_shape_3d, device=h.device)
                inner_state = (m_inner, v_inner, prev_out_inner)
            else:
                inner_state = None
            
            momentum = (outer_state, inner_state)
        elif "e" in self.arch:
            momentum = (
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.zeros(1, device = h.device, dtype = torch.long),
                torch.zeros_like(h),
            )
        elif "a" in self.arch:
            momentum = (
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.zeros_like(h),
                )
        elif "r" in self.arch or "l" in self.arch or "o" in self.arch or "v" in self.arch:
            momentum = (
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.zeros_like(h),
                torch.zeros_like(h),
            )
        else: # in case of no momentum --mu will be set to zero
            momentum = torch.zeros_like(h)
        
        for i, layer in enumerate(self.layers):
            if layer.use_attn:
                cache_size = layer.attn.attn.get_cache_size()
                if cache_size > block_size:
                    h_cache_next_l = torch.cat(
                        [h_cache[i][:, -cache_size + block_size:, :], h],
                        dim = 1
                    ).detach()
                else:
                    h_cache_next_l = h[:, -cache_size:, :].detach()
                h_cache_next.append(h_cache_next_l)
                h, momentum = layer(h, h_cache[i], self.pos_encoding, momentum) # B x M x H
            else:
                # TODO: is this branch even necesarry in our case?
                h = layer(h, [], self.pos_encoding)
        
        out = F.log_softmax(self.out_embed(h), dim = -1)
        return out, h_cache_next
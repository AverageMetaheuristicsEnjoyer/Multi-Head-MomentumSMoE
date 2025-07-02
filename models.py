import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_transformer import FMoETransformerMLP
from gates import CustomNaiveGate_Balance_SMoE, MHMoEGate

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
        # self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, inp, momentum):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)

        momentum = self.mu * momentum + self.gamma * moe_out
        # output = self.layer_norm(inp - momentum)
        output = inp - momentum
        return output, momentum

class AdamLayer(FMoETransformerMLP):
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

    def forward(self, inp, momentum):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)
        
        if self.layerth == 0:
            p = momentum[0]
            v = momentum[1]
            momentum = self.mu * momentum[2] + self.gamma2 * moe_out

            p = self.beta1 * p + (1 - self.beta1) * moe_out
            v = self.beta2 * v + (1 - self.beta2) * (moe_out ** 2)
            adam = (self.gamma1 / torch.sqrt(v + 1e-8)) * p + inp
            output = inp - adam  
        else:
            p = momentum[0]
            v = momentum[1]
            momentum = self.mu * momentum[2] + self.gamma2 * moe_out
            # output = self.layer_norm(inp - momentum)
            output = inp - momentum
        
        return output, (p, v, momentum)

def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0-a) * alpha_start + a * alpha_end
    return alpha_end

def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5)/math.log(beta+eps)-1

    def f_inv(t):
        return math.pow(0.5, 1/(t+1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
    return beta_end

class AdEMAMixLayer(FMoETransformerMLP):
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
        use_xmoe,
        xmoe_dim,
        world_size,
        weight_decay,
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
        self.mu = mu
        self.gamma2 = gamma2
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        # self.t_warmup = t_warmup
        self.alpha_warmup = alpha_warmup
        self.beta3_warmup = beta3_warmup
        self.ademamix_all_layers = ademamix_all_layers
        self.weight_decay = weight_decay
        self.layerth = layerth
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, momentum):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)

        if self.layerth == 0 or self.ademamix_all_layers:
            m1, v, m2, step_count, _ = momentum
            step_count += 1
            step = step_count.item()
            bias_correction1 = 1.0 - self.beta1 ** step
            bias_correction2 = 1.0 - self.beta2 ** step

            if self.alpha_warmup:
                alpha_t = linear_warmup_scheduler(step, alpha_end = self.alpha, alpha_start = 0, warmup = self.alpha_warmup)
            else:
                alpha_t = self.alpha
            
            if self.beta3_warmup:
                beta3_t = linear_hl_warmup_scheduler(step, self.beta3, beta_start = self.beta1, warmup = self.beta3_warmup)
            else:
                beta3_t = self.beta3

            m1_new = self.beta1 * m1 + (1 - self.beta1) * moe_out
            v_new = self.beta2 * v + (1 - self.beta2) * (moe_out ** 2)
            m2_new = beta3_t * m2 + (1 - beta3_t) * moe_out
            momentum = self.mu * momentum[4] + self.gamma2 * moe_out
            
            denom = torch.sqrt(v_new / bias_correction2 + 1e-8)
            update = (m1_new / bias_correction1 + alpha_t * m2_new) / denom
            
            if self.weight_decay > 0:
                update = update + self.weight_decay * inp
                
            output = inp - update
            
            return output, (m1_new, v_new, m2_new, step_count, momentum)
        else:
            m1, v, m2, step_count, _ = momentum
            momentum = self.mu * momentum[4] + self.gamma2 * moe_out
            
            output = inp - momentum
            return output, (m1, v, m2, step_count, momentum)

class SignumLayer(FMoETransformerMLP):
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
        rand_zero,
        use_xmoe,
        xmoe_dim,
        world_size,
        beta1,
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
        self.beta1 = beta1
        self.rand_zero = rand_zero
        self.layerth = layerth
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, momentum):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)
        
        v = momentum
        moe_out = torch.sign(moe_out)
        if self.rand_zero:
            is_zero = (moe_out == 0)
            moe_out[is_zero] = 2 * torch.random.randint(0, 2, size = torch.sum(is_zero)) - 1
        signum = self.beta1 * v + (1 - self.beta1) * moe_out
        output = inp - signum
        
        return output, signum

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

    def forward(self, inp, hist):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)
        
        m, v, step_count, moe_out_prev = hist
        step_count += 1
        # step = step_count.item()
        # bias_correction_m = 1.0 - self.beta1 ** step
        # bias_correction_v = 1.0 - self.beta2 ** step

        c = moe_out + self.gamma2 * (self.beta1 / (1 - self.beta1)) * (moe_out - moe_out_prev)
        c_norm = torch.linalg.matrix_norm(c, dim = (-2, -1), ord = "fro")
        batch_idx = c_norm > 1
        scaling_facs = torch.ones_like(c_norm)
        scaling_facs[batch_idx] = c_norm[batch_idx]
        c_t = c / scaling_facs.view(-1, 1, 1)
        
        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2

        # m_t /= bias_correction_m
        # v_t /= bias_correction_v

        out = inp - self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))

        return out, (m_t, v_t, step_count, moe_out)
    
class MarsLionLayer(FMoETransformerMLP):
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

    def forward(self, inp, hist):
        moe_out = super().forward(inp)
        moe_out = self.dropout(moe_out)
        
        m, v, step_count, moe_out_prev = hist

        c = moe_out + self.gamma2 * (self.beta1 / (1 - self.beta1)) * (moe_out - moe_out_prev)
        c_norm = torch.linalg.matrix_norm(c, dim = (-2, -1), ord = "fro")
        batch_idx = c_norm > 1
        scaling_facs = torch.ones_like(c_norm)
        scaling_facs[batch_idx] = c_norm[batch_idx]
        c_t = c / scaling_facs.view(-1, 1, 1)
        
        m_t = self.beta1 * m + (1 - self.beta1) * c_t

        out = inp - self.gamma1 * torch.sign(m_t)

        return out, (m_t, v, step_count, moe_out)

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

        c = (1 + self.gamma2) * moe_out - self.gamma2 * moe_out_prev
        m_t = m
        v_t = v

        out = inp - c

        return out, (m_t, v_t, momentum, moe_out)

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

    def forward(self, inp, hist):
        moe_out = super().forward(self.layer_norm(inp))
        moe_out = self.dropout(moe_out)
        
        m, v, step_count, moe_out_prev = hist
        step_count += 1

        c_t = (1 + self.gamma2) * moe_out - self.gamma2 * moe_out_prev
        m_t = self.beta1 * m + (1 - self.beta1) * c_t
        v_t = self.beta2 * v + (1 - self.beta2) * c_t**2

        out = inp - self.gamma1 * m_t / (torch.sqrt(v_t + 1e-8))

        return out, (m_t, v_t, step_count, moe_out)

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
    ):
        super().__init__()
        if gate_name == "smoe":
            gate = CustomNaiveGate_Balance_SMoE # from SwitchTransformer paper
        elif gate_name == "mhmoe":
            gate = MHMoEGate
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
        
        self.use_smoe = g in ["m", "a", "e", "r", "l", "o", "v"]
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
            AdamLayer(
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
            if g == "a"
            else
            AdEMAMixLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
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
                world_size = world_size,
                layerth = layerth,
            )
            if g == "e"
            else
            SignumLayer(
                hidden_size = hidden_size,
                inner_hidden_size = inner_hidden_size,
                dropout = dropout,
                gate = gate,
                num_experts = num_experts,
                moe_top_k = moe_top_k,
                mhmoe_num_heads = mhmoe_num_heads,
                mhmoe_beta = mhmoe_beta,
                use_xmoe = use_xmoe,
                xmoe_dim = xmoe_dim,
                world_size = world_size,
                beta1 = beta1,
                rand_zero = rand_zero,
                layerth = layerth,
            )
            if g == "u"
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
            )
            if g == "r"
            else
            MarsLionLayer(
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
            if g == "l"
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
                layerth = i
            )
            for i in range(num_layers)
        )
    
    def forward(self, x, h_cache):
        block_size = x.size(1) # B x M
        h = self.inp_embed(x) # B x M x H
        h_cache_next = []
        if "e" in self.arch:
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

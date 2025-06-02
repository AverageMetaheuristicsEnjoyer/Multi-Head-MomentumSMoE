import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None

class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

def _one_hot_with_dtype(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result

class MHMoEGate(BaseGate):
    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        top_k = 2,
        use_xmoe = False,
        xmoe_routing_dim = 8,
    ):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.loss = None
        self.use_xmoe = use_xmoe
        self.xmoe_routing_dim = xmoe_routing_dim
        if self.use_xmoe:
            self.wg_reduction = nn.Linear(d_model, xmoe_routing_dim, bias = False)
            wg = torch.empty(num_expert, xmoe_routing_dim)
            nn.init.orthogonal_(wg, gain = 0.32)
            self.register_parameter("wg", nn.Parameter(wg))
        else:
            self.wg = nn.Linear(d_model, num_expert, bias = False)
        
    
    def _cosine(self, mat1, mat2, eps = 1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p = 2.0, dim = 1, eps = eps)
        mat2 = F.normalize(mat2.float(), p = 2.0, dim = 1, eps = eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
    
    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

    def _calculate_load_balance_loss(self, gate, top_ids):
        scores_w_noise = F.softmax(gate / 0.3, dim=-1)
        num_samples, num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
        mask = _one_hot_with_dtype(
            top_ids[:, 0],
            num_global_experts,
            dtype = scores_w_noise.dtype,
            hot_value = num_global_experts / num_samples
        )
        me = torch.sum(scores_w_noise, dim = 0)
        ce = torch.sum(mask, dim = 0)
        self.loss = torch.sum(me * ce) / num_samples
    
    def forward(self, inp, return_all_scores = False):
        if self.use_xmoe:
            inp = self.wg_reduction(inp)
            with torch.no_grad():
                wg_norm = self.wg.norm(p = 2.0, dim = -1, keepdim = True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(inp, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(inp)

        gate_top_k_logits, gate_top_k_idx = torch.topk(
            logits,
            k = self.top_k,
            dim = -1,
            largest = True,
            sorted = False,
        )

        gate_score = F.softmax(gate_top_k_logits, dim = -1)
        self._calculate_load_balance_loss(logits, gate_top_k_idx)
        
        if return_all_scores:
            return gate_top_k_idx, gate_score, logits
        return gate_top_k_idx, gate_score
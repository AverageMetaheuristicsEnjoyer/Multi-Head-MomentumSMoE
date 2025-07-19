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
    def __init__(self, d_model, num_expert, world_size, top_k=2, aux_blance=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias=False)
        self.top_k = top_k
        self.aux_blance = aux_blance
        self.loss = None

    def _calculate_load_balance_loss(self, router_probs, top_k_indices):
        with torch.no_grad():
            one_hot_indices = F.one_hot(top_k_indices, self.tot_expert).float()
            one_hot_indices = torch.sum(one_hot_indices, dim = 1)
            f_i = one_hot_indices.mean(dim=0)

        P_i = torch.mean(router_probs.float(), dim=0)

        loss = (f_i * P_i).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):
        logits = self.gate(inp)

        top_k_logits, top_k_indices = torch.topk(
            logits, k=self.top_k, dim=-1, largest=True, sorted=False
        )

        router_probs = torch.full_like(logits, float("-inf"))
        router_probs.scatter_(-1, top_k_indices, top_k_logits)
        router_probs = F.softmax(router_probs, dim=-1)
        if self.training and self.aux_blance:
            self._calculate_load_balance_loss(router_probs, top_k_indices)
        
        top_k_scores = torch.gather(router_probs, dim = -1, index = top_k_indices)

        if return_all_scores:
            return top_k_indices, top_k_scores, logits
        return top_k_indices, top_k_scores

class SMoE_Momentum(BaseGate):
    def __init__(self, d_model, num_expert, world_size,
                 top_k=2, aux_blance=True, smome_alpha = 1.0, smome_beta = 0.9, **kwargs):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias=False)
        self.top_k = top_k
        self.aux_blance = aux_blance
        self.loss = None
        self.register_buffer("avg_logits", torch.zeros(self.tot_expert))
        self.alpha = smome_alpha
        self.beta = smome_beta

    def forward(self, inp, return_all_scores=False):
        logits = self.gate(inp)

        if self.training:
            penalty = self.avg_logits.unsqueeze(0) * self.alpha
            balanced_logits = logits - penalty

            mean_batch_logits = torch.mean(logits.float(), dim = 0)
            with torch.no_grad():
                self.avg_logits.mul_(self.beta)
                
                self.avg_logits.add_(mean_batch_logits.detach(), alpha = 1.0 - self.beta)
        else:
            balanced_logits = logits

        top_k_logits, top_k_indices = torch.topk(
            balanced_logits, k=self.top_k, dim=-1, largest=True, sorted=False
        )
        
        router_probs = torch.full_like(balanced_logits, float("-inf"))
        router_probs.scatter_(-1, top_k_indices, top_k_logits)
        router_probs = F.softmax(router_probs, dim=-1)
        
        top_k_scores = torch.gather(router_probs, dim = -1, index = top_k_indices)

        if return_all_scores:
            return top_k_indices, top_k_scores, logits
        return top_k_indices, top_k_scores

# class EF21Gate(BaseGate):
#     def __init__(self, d_model, num_expert, world_size, top_k=2):
#         super().__init__(num_expert, world_size)
#         self.gate = nn.Linear(d_model, self.tot_expert)
#         self.top_k = top_k
#         self.top_kc = 1
#         self.loss = None

#         self.register_buffer('g_t', None)

#     def forward(self, inp, g_t):
#         gate = self.gate(inp)

#         if self.g_t is None or self.g_t.shape != gate.shape:
#             initial_top_k_val, initial_top_k_idx = torch.topk(
#                 gate, k=self.top_k, dim=-1, largest=True, sorted=False
#             )
#             self.g_t = torch.zeros_like(gate)
#             self.g_t.scatter_(dim=-1, index=initial_top_k_idx, src=initial_top_k_val)


#         gate = gate - self.g_t.detach()

#         gate_top_k_val, gate_top_k_idx = torch.topk(
#             gate, k=self.top_k, dim=-1, largest=True, sorted=False
#         )

#         c_t = torch.zeros_like(gate)
#         c_t = c_t.scatter_(
#             dim = -1,
#             index = gate_top_k_idx,
#             src = gate_top_k_val,
#         )
#         new_g_t = self.g_t + c_t

#         gate_top_k_val, gate_top_k_idx = torch.topk(
#             new_g_t, k=self.top_k, dim=-1, largest=True, sorted=False
#         )

#         gate_score = F.softmax(gate_top_k_val.view(-1, self.top_k), dim=-1)

#         self.g_t = new_g_t.detach()

#         return gate_top_k_idx, gate_score, new_g_t


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
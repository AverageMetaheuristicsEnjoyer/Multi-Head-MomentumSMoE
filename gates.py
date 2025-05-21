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

class MHMoEGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k = 2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.loss = None
    
    def _calculate_load_balance_loss(self, routing_weights):
        sub_tokens_count = routing_weights.size(0)
        if sub_tokens_count == 0:
            self.loss = torch.tensor(0.0, device = routing_weights.device)
            return

        top_k_weights, top_k_idx = torch.topk(
            routing_weights,
            k = self.top_k,
            dim = -1,
        )
        
        gates = torch.zeros_like(routing_weights)
        gates = gates.scatter_(-1, top_k_idx, F.softmax(top_k_weights, dim = -1))

        expert_counts = gates.sum(0)
        router_freq = gates.mean(0)

        self.loss = self.tot_expert * (router_freq * (expert_counts / sub_tokens_count)).sum()
    
    def forward(self, inp, return_all_scores = False):
        routing_weights = self.gate(inp)

        gate_top_k_logits, gate_top_k_idx = torch.topk(
            routing_weights,
            k = self.top_k,
            dim = -1,
            largest = True,
            sorted = False,
        )

        gate_score = F.softmax(gate_top_k_logits, dim = -1)
        self._calculate_load_balance_loss(routing_weights)
        
        if return_all_scores:
            return gate_top_k_idx, gate_score, routing_weights
        return gate_top_k_idx, gate_score
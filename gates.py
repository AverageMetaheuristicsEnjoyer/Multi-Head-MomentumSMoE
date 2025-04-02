import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    def __init__(self, d_model, num_experts, top_k = 2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts
        
    def forward(self, x):
        routing_weights = self.gate(x)
        
        top_k_weights, top_k_indices = torch.topk(
            routing_weights,
            k = self.top_k,
            dim = -1,
        )

        gates = torch.zeros_like(routing_weights)
        gates = gates.scatter_(-1, top_k_indices, F.softmax(top_k_weights, dim = -1))
        return gates, routing_weights
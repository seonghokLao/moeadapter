import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class LoRA_MoElayer(nn.Module):
    def __init__(self, dim, lora_dim=[8, 16, 32, 48, 64, 96, 128], noisy_gating=True, k=1):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.k = k
        self.num_experts = len(lora_dim)

        self.Lora_a_experts = nn.ModuleList([nn.Linear(dim, d, bias=False) for d in lora_dim])
        self.Lora_b_experts = nn.ModuleList([nn.Linear(d, dim, bias=False) for d in lora_dim])

        for a in self.Lora_a_experts:
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in self.Lora_b_experts:
            nn.init.zeros_(b.weight)

        self.w_gate = nn.Parameter(torch.zeros(dim, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim, self.num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        nn.init.normal_(self.w_noise, mean=0.0, std=0.02)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        eps = 1e-6
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)

        clean_values = torch.nan_to_num(clean_values, nan=0.0)
        noisy_values = torch.nan_to_num(noisy_values, nan=0.0)
        noise_stddev = torch.nan_to_num(noise_stddev, nan=1.0, posinf=1e2, neginf=1e-2)

        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        noise_stddev = noise_stddev.clamp(min=eps)

        logits_if_in = (clean_values - threshold_if_in) / noise_stddev
        logits_if_out = (clean_values - threshold_if_out) / noise_stddev

        logits_if_in = logits_if_in.clamp(min=-10, max=10)
        logits_if_out = logits_if_out.clamp(min=-10, max=10)

        prob_if_in = normal.cdf(logits_if_in)
        prob_if_out = normal.cdf(logits_if_out)

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate

        if torch.isnan(clean_logits).any():
            print("NaNs in clean_logits!")
            print("x stats:", x.mean().item(), x.std().item())
            print("w_gate stats:", self.w_gate.mean().item(), self.w_gate.std().item())

        if self.noisy_gating and train:
            raw_noise_stddev = torch.clamp(x @ self.w_noise, min=-5.0, max=5.0)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = torch.nan_to_num(noisy_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            logits = torch.nan_to_num(clean_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        if torch.isnan(x).any():
            print("Nans found in MoE input x")

        gates, load = self.noisy_top_k_gating(x, self.training)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            qkv_delta = F.linear(expert_inputs[i], self.Lora_a_experts[i].weight)
            qkv_delta = F.linear(qkv_delta, self.Lora_b_experts[i].weight)
            expert_outputs.append(qkv_delta)

        y = dispatcher.combine(expert_outputs)
        y = y.reshape(B, N, C)

        # Final safety net
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        y = torch.clamp(y, min=-1e3, max=1e3)

        return y, loss


import math  # ensure this is at the end or top of the file




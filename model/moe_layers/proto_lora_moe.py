import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from collections import deque



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class LoRA_MoElayer(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, dim, lora_dim=[8,16,32,48,64,96,128], noisy_gating=True, k=1): #
        super(LoRA_MoElayer, self).__init__()

        self.noisy_gating = noisy_gating
        self.k = k
        self.momentum = 0.999
        self.kappa = 0.1

        # instantiate lora experts
        Lora_a_experts = nn.ModuleList()
        Lora_b_experts = nn.ModuleList()
        for i,d in enumerate(lora_dim):
            Lora_a_experts.append(nn.Linear(dim, d,bias = False))
            nn.init.kaiming_uniform_(Lora_a_experts[i].weight, a=math.sqrt(5))
            Lora_b_experts.append(nn.Linear(d, dim,bias = False))
            nn.init.zeros_(Lora_b_experts[i].weight)

        # define lora param
        self.num_experts = len(Lora_a_experts)
        self.Lora_a_experts = Lora_a_experts
        self.Lora_b_experts = Lora_b_experts

        self.prototypes = nn.Parameter(torch.randn(self.num_experts, dim), requires_grad=False)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))


        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        eps = 1e-6
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev.clamp(min=eps))
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev.clamp(min=eps))
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def sinkhorn(self, log_alpha, num_iters=3, epsilon=1e-6):
        """
        log_alpha: [K, N] (log of similarity matrix or cost matrix, usually negative distances)
        Returns doubly-stochastic matrix L: [K, N]
        """
        # K, N = log_alpha.shape
        # u = torch.zeros(K, device=log_alpha.device)
        # v = torch.zeros(N, device=log_alpha.device)

        L = torch.exp(log_alpha)
        for _ in range(num_iters):
            L = L / (L.sum(dim=1, keepdim=True) + epsilon)
            L = L / (L.sum(dim=0, keepdim=True) + epsilon)
        
        return L

    def sinkhorn_routing(self, x):

        """
        x: [B, D] inputs
        prototypes: [K, D] expert centroids
        """
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        similarity = torch.matmul(p_norm, x_norm.T)
        log_alpha = similarity / self.kappa

        L = self.sinkhorn(log_alpha)
        return L.T
    
    def update_prototypes(self, x, gates):
        for i in range(self.num_experts):
            weights = gates[:, i].unsqueeze(1)
            if weights.sum() > 0:
                proto_update = (weights * x).sum(0) / weights.sum()
                self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * proto_update

    def forward(self, x, loss_coef=1):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        B, N, C = x.shape
        x_mean = torch.mean(x, dim=1, keepdim=False)

        with torch.no_grad():
            gates = self.sinkhorn_routing(x_mean.detach())

            self.update_prototypes(x_mean, gates)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_mean)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            out = F.linear(F.gelu(F.linear(expert_inputs[i], self.Lora_a_experts[i].weight)), self.Lora_b_experts[i].weight)
            expert_outputs.append(out)

        y = dispatcher.combine(expert_outputs)
        y = y.unsqueeze(1)
        loss = 0
        return y, loss


import math
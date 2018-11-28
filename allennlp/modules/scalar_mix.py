from typing import List

import torch
from torch.nn import ParameterList, Parameter

from allennlp.common.checks import ConfigurationError

class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """
    def __init__(self,
                 mixture_size: int,
                 do_layer_norm: bool = False,
                 initial_scalar_parameters: List[float] = None,
                 num_heads: int = None, use_temp: bool = False,
                 trainable: bool = True,
                 apply_softmax: bool = True) -> None:
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.num_heads = num_heads

        if num_heads is None:
            num_heads = 1
        else:
            assert not do_layer_norm

        self.num_heads = num_heads

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ConfigurationError("Length of initial_scalar_parameters {} differs "
                                     "from mixture_size {}".format(
                                             initial_scalar_parameters, mixture_size))

        self.scalar_parameters = ParameterList(
                [Parameter(torch.FloatTensor([initial_scalar_parameters[i]] * num_heads),
                           requires_grad=trainable) for i
                 in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)
        if use_temp:
            self.scalar_temp = Parameter(torch.FloatTensor([1.0]))
        self.use_temp = use_temp
        self.apply_softmax = apply_softmax
        if not self.apply_softmax:
            assert num_heads == 1
            assert not self.use_temp

    def forward(self, tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ConfigurationError("{} tensors were passed, but the module was initialized to "
                                     "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1E-12)

        if self.use_temp:
            zero = torch.FloatTensor([0.0]).to(self.scalar_temp.device)
            temp = torch.max(self.scalar_temp, zero)
        else:
            temp = 1.0

        if self.num_heads == 1:
            if self.apply_softmax:
                normed_weights = torch.nn.functional.softmax(temp * torch.cat([parameter for parameter
                                                                in self.scalar_parameters]), dim=0)
            else:
                normed_weights = torch.clamp(torch.cat([parameter for parameter in self.scalar_parameters]), min=0.0, max=1e12)

            normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        else:
            # softmax normalize across layers for each head
            normed_weights = torch.nn.functional.softmax(
                    temp * torch.cat([parameters.unsqueeze(0) for parameters in self.scalar_parameters], dim=0),
            dim=0)
            normed_weights = [
                    w.squeeze(0)
                    for w in torch.chunk(normed_weights, dim=0, chunks=self.mixture_size)
            ]

        if not self.do_layer_norm:

            if self.num_heads == 1:
                pieces = []
                for weight, tensor in zip(normed_weights, tensors):
                    pieces.append(weight * tensor)
                return self.gamma * sum(pieces)
            else:
                head_dim = tensors[0].shape[-1] // self.num_heads
                pieces = []
                for weight, tensor in zip(normed_weights, tensors):
                    # need to partition last dimension of tensor
                    # heads = list length num_layers each with one head
                    heads = torch.split(tensor, split_size_or_sections=head_dim, dim=-1)
                    weighted_heads = []
                    for w, h in zip(
                            torch.split(weight, split_size_or_sections=1),
                            heads
                    ):
                        weighted_heads.append(w * h)

                    pieces.append(torch.cat(weighted_heads, dim=-1))

                return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)

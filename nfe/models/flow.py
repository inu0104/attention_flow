from typing import List, Optional

import torch.nn as nn
import stribor as st
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, MultiheadAttention
from torch.fft import fft, ifft


class CouplingFlow(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        transforms = []
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
                time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))

        self.flow = st.Flow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        t0: Optional[Tensor] = None,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]

        return self.flow(x, t=t)[0]


class ResNetFlow(Module):
    """
    ResNet flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the residual neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        invertible: Whether to make ResNet invertible (necessary for proper flow)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        invertible: Optional[bool] = True,
        **kwargs
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(st.net.ResNetFlow(
                dim,
                hidden_dims,
                n_layers,
                activation='ReLU',
                final_activation=None,
                time_net=time_net,
                time_hidden_dim=time_hidden_dim,
                invertible=invertible
            ))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        for layer in self.layers:
            x = layer(x, t)

        return x


class FourierFlow(Module):
    """
    Fourier Transform-based Flow model

    Args:
        dim: Data dimension
        n_layers: Number of Fourier layers
        hidden_dim: Hidden dimensions of the transformation
    """
    def __init__(self, dim, n_layers, hidden_dim):
        super().__init__()
        self.layers = ModuleList([
            nn.Sequential(
                Linear(dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, dim)
            ) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        # Apply Fourier transform
        fourier_coeff = fft(x, dim=-2).real  # Only real part
        for layer in self.layers:
            fourier_coeff = layer(fourier_coeff)

        return ifft(fourier_coeff, dim=-2).real  # Return to time domain
    
    
class AttentionFlow(Module):
    """
    Attention-based Flow model with input dimension adjustment.

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dim: Hidden dimension for intermediate layers (must match embed_dim)
        n_heads: Number of attention heads
    """
    def __init__(self, dim, n_layers, hidden_dim, n_heads):
        super().__init__()

        # Linear layers to adjust input/output dimensions
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dim)

        # Attention layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # Project input to match embed_dim
        x = self.input_proj(x)

        # Ensure time steps match
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        # Apply attention layers
        for layer in self.layers:
            attn_output, _ = layer(x, x, x)  # Query, Key, Value are all x
            x = attn_output

        # Project back to original dimension
        x = self.output_proj(x)
        return x


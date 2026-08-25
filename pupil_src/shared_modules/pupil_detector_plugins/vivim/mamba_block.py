"""
Selective State Space Model (SSM) Layer wrapping the official Mamba-3 module.
Uses mamba_ssm.modules.mamba3.Mamba3 directly from the state-spaces/mamba package.
"""

import torch
import torch.nn as nn

# Monkeypatch missing low-precision float types in PyTorch <= 2.6.0
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = "dummy_float4_e2m1fn_x2"
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = "dummy_float8_e8m0fnu"

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except (ImportError, AttributeError):
    Mamba3 = None


class MambaLayer(nn.Module):
    """
    Drop-in replacement for MambaLayer.
    Interface contract: forward(x: [B, L, D]) -> [B, L, D]
    All internal SSM logic is delegated to the official Mamba3 module.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba3.Mamba3 could not be imported. "
                "Ensure mamba-ssm is installed in the active Python environment."
            )
        self.mamba3 = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        return self.mamba3(x)

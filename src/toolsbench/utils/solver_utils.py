"""Solver utilities shared across PnP solvers.

This module provides helpers for step size computation, reconstruction
initialization, and normalization strategies, used by both single-GPU
and distributed PnP solvers.
"""

import torch
import copy


def compute_step_size_from_operator(operator, ground_truth: torch.Tensor) -> float:
    """Compute PnP step size as 1 / Lipschitz constant of the forward operator.

    Parameters
    ----------
    operator : deepinv.physics.Physics
        Physics operator (can be stacked or distributed).
    ground_truth : torch.Tensor
        Ground truth tensor used to create an example signal for norm computation.

    Returns
    -------
    float
        Step size = 1 / lipschitz_constant, or 1.0 if constant is non-positive.
    """
    with torch.no_grad():
        x_example = torch.zeros_like(
            ground_truth, device=ground_truth.device, dtype=ground_truth.dtype
        )
        lipschitz_constant = operator.compute_norm(x_example, local_only=False)
        return 1.0 / lipschitz_constant if lipschitz_constant > 0 else 1.0


def initialize_reconstruction(
    signal_shape: tuple,
    operator,
    measurements,
    device: torch.device,
    method: str = "adjoint",
    clip_range: tuple = None,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Initialize the reconstruction signal.

    Parameters
    ----------
    signal_shape : tuple
        Shape of the signal to initialize.
    operator : deepinv.physics.Physics
        Physics operator (can be stacked or distributed).
    measurements : torch.Tensor or TensorList
        Observed measurements.
    device : torch.device
        Device to create the tensor on.
    method : str, optional
        Initialization method:
        - ``"zeros"``: start from zero (always safe; works for any physics).
        - ``"pseudo_inverse"``: ``x_0 = A†y`` clamped to ``[0, 1]`` (natural
          images / bounded domains).
        - ``"adjoint"``: ``x_0 = Aᵀy`` without clamping (radio, tomography,
          or any unbounded physical domain).
    clip_range: (sig_min, sig_max) for scaling the dirty image
    weights: Optional density weights for weighted dirty image

    Returns
    -------
    torch.Tensor
        Initialized reconstruction tensor on ``device``.
    """
    if method == "zeros":
        return torch.zeros(signal_shape, device=device)

    elif method == "pseudo_inverse":
        if weights is not None:
            dirty = operator.A_dagger(measurements*weights.to(device))
        else:
            dirty = operator.A_dagger(measurements)

        if clip_range is not None:
            sig_min, sig_max = clip_range
            # Peak-normalize dirty image to signal range and clamp
            x_init = dirty * sig_max / dirty.max()
            x_init = x_init.clamp(sig_min, sig_max)
        return x_init

    elif method == "adjoint":
        if weights is not None:
            dirty = operator.A_adjoint(measurements*weights.to(device))
        else:
            dirty = operator.A_adjoint(measurements)

        if clip_range is not None:
            sig_min, sig_max = clip_range
            # Peak-normalize dirty image to signal range and clamp
            x_init = dirty * sig_max / dirty.max()
            x_init = x_init.clamp(sig_min, sig_max)
        return x_init

    else:
        raise ValueError(
            f"Unknown initialization method: '{method}'. "
            "Choose from 'zeros', 'pseudo_inverse', or 'adjoint'."
        )


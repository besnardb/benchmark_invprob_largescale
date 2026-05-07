import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import constants as const
from deepinv.physics import RadioInterferometry, LinearPhysics
from typing_extensions import TypeAlias
import pytorch_finufft as py_nufft

FilePathType: TypeAlias = Union[Path, str]
DEFAULT_DEVICE = torch.device("cpu")

class MyRadioInterferometry(LinearPhysics):
    """Benchmark-compatible radio interferometry operator.

    This mirrors ``toolsbench.utils.deepinv_imager.MyRadioInterferometry``:
    the NUFFT is intentionally called with ``norm=None`` to preserve the
    benchmark radio scaling.
    """

    def __init__(
        self,
        img_size: tuple[int, int] | torch.Tensor,
        samples_loc: torch.Tensor,
        dataWeight: torch.Tensor | None = None,
        real_projection: bool = True,
        device: torch.device | str = "cpu",
        **kwargs,
    ) -> None:

        super().__init__(device=device, **kwargs)

        if isinstance(img_size, torch.Tensor):
            img_size = tuple(int(v) for v in img_size.detach().cpu().tolist())
        self.img_size = tuple(int(v) for v in img_size)
        if len(self.img_size) != 2:
            raise ValueError(f"Expected 2-D image size, got {self.img_size}.")

        if dataWeight is None:
            dataWeight = torch.tensor([1.0], device=device)

        self.real_projection = bool(real_projection)

        self.register_buffer("samples_loc", samples_loc.to(device))
        self.register_buffer("dataWeight", dataWeight.to(device))

        if self.real_projection:
            self.adj_projection = lambda x: torch.real(x).to(torch.float)
        else:
            self.adj_projection = lambda x: x

        self.to(device)

    def setWeight(self, w: torch.Tensor) -> None:
        self.dataWeight = w.to(self.dataWeight)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return py_nufft.functional.finufft_type2(self.samples_loc, x.to(torch.cfloat), modeord=0, isign=-1) * self.dataWeight

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.adj_projection(py_nufft.functional.finufft_type1(self.samples_loc, y * self.dataWeight, self.img_size, modeord=0, isign=1))

@dataclass
class DirtyImagerConfig:
    """Base class for the config / parameters of a dirty imager.

    Contains basic parameters common across all dirty imagers.
    Inherit and add parameters specific to a dirty imager implementation.

    Attributes:
        imaging_npixel (int): Image size
        imaging_cellsize (float): Scale of a pixel in radians
        combine_across_frequencies (bool): Whether or not to combine images
            across all frequency channels into one image. Defaults to True.

    """

    imaging_npixel: int
    imaging_cellsize: float
    binning_factor: float = 1.25
    nufft_k_oversampling: float = 1.5
    combine_across_frequencies: bool = True


class DeepinvDirtyImager(torch.nn.Module):
    """Dirty imager based on the DeepInv library.

    Attributes:
        config (DirtyImagerConfig): Config containing parameters for
            dirty imaging

    """

    def __init__(
        self, config: DirtyImagerConfig, device=torch.device("cpu"), verbose: int = 0
    ) -> None:
        """Initializes the instance with a config.

        Args:
            config (DirtyImagerConfig): see config attribute
            device (torch.device): Device to use for computations. Default is torch.device("cpu").

        """
        super().__init__()
        self.config = config
        self.device = device  # setup_device()  # Ensure device is set up correctly
        self.verbose = verbose

    def to_device(self, tensor, non_blocking=True, pin_memory=False):
        """Transfer tensor to device with optimized settings and error handling."""
        try:
            if self.device.type == "cuda":
                # Pin memory for faster transfers if specified
                if pin_memory and tensor.device.type == "cpu":
                    tensor = tensor.pin_memory()
                return tensor.to(self.device, non_blocking=non_blocking)
            else:
                return tensor.to(self.device)
        except RuntimeError as e:
            print(f"Transfer error to {self.device}: {e}")
            print("Falling back to CPU")
            return tensor.to("cpu")

    def load_visibilities(
        self,
        visibility_path: str,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load data from MS file and convert to PyTorch tensors

        Args:
            visibility_path (str): Path to the visibility data file
            visibility_format (str): Format of the visibility data. Currently only "MS" is supported.
            visibility_column (str): Column name in the MS file containing the visibility data. Default is for "DATA". for the OSKAR simulator.

        """
        from casacore.tables import table

        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} not supported, "
                "only MS format is currently supported"
            )
        # Get UVW coords and visibilities
        with table(visibility_path, readonly=True) as tb:

            # Direct loading with proper type
            uvw_np = tb.getcol("UVW").astype(np.float32)
            visibilities_np = tb.getcol(visibility_column).astype(np.complex64)

            print(f"Data loaded: {uvw_np.shape[0]} visibilities")
            print(f"Available columns: {tb.colnames()}")

        # Load frequencies
        with table(visibility_path + "/SPECTRAL_WINDOW", readonly=True) as tb:
            chan_freqs_np = tb.getcol("CHAN_FREQ")[0].astype(np.float32)

        print(f"Number of channels: {len(chan_freqs_np)}")

        # Optimized conversion to PyTorch with direct transfer to device
        uvw = self.to_device(torch.from_numpy(uvw_np))
        visibilities = self.to_device(torch.from_numpy(visibilities_np))
        freqs = self.to_device(torch.from_numpy(chan_freqs_np))

        # CPU memory cleanup
        del uvw_np, visibilities_np, chan_freqs_np

        return uvw, visibilities, freqs

    def normalize_uv_coords(
        self,
        uvw: torch.Tensor,
        freqs: torch.Tensor,
        visibilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize UV coordinates and concatenate visibilities for NUFFT

        Optimizations:
        - Vectorized calculations
        - Tensor pre-allocation
        - Use of pre-calculated constants
        """

        # Stokes I polarization (vectorized)
        if visibilities.shape[2] == 4:
            visibilities = 0.5 * (visibilities[:, :, 0] + visibilities[:, :, 3])
        elif visibilities.shape[2] == 1:
            visibilities = visibilities[:, :, 0]

        n_vis, n_freq = visibilities.shape
        im_size = torch.tensor(
            [self.config.imaging_npixel, self.config.imaging_npixel], device=self.device
        )

        # Pre-allocation for better performance
        total_points = n_vis * n_freq
        samples_locs = torch.zeros(
            (2, total_points), dtype=torch.float32, device=self.device
        )
        all_visibilities = torch.zeros(
            total_points, dtype=torch.complex64, device=self.device
        )

        # Vectorized calculations
        uv_base = uvw[:, :2]  # [n_vis, 2]
        cellsize_2pi = self.config.imaging_cellsize * 2 * np.pi

        # Processing by frequency (more memory efficient)
        for i, freq in enumerate(freqs):
            start_idx = i * n_vis
            end_idx = (i + 1) * n_vis

            # Vectorized calculation of normalized UV coordinates
            wavelength = const.c.value / freq
            uv_lambda = uv_base / wavelength
            uv_norm = (uv_lambda * cellsize_2pi).T
            uv_norm = torch.stack((-uv_norm[1], uv_norm[0]), dim=0)

            samples_locs[:, start_idx:end_idx] = uv_norm
            all_visibilities[start_idx:end_idx] = visibilities[:, i]

        # Reshape for compatibility with rest of code
        visibilities_reshaped = all_visibilities.unsqueeze(0).unsqueeze(0)

        return samples_locs, visibilities_reshaped

    @staticmethod
    def uniform_weighting(
        u: torch.Tensor,
        v: torch.Tensor,
        im_size: torch.Tensor,
        weight_gridsize: int = 1,
        device=DEFAULT_DEVICE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strict uniform weighting: w_i = 1 / n(cell_i).
        """

        dtype = torch.float32
        N0, N1 = [int(i * weight_gridsize) for i in im_size]

        # UV symmetrization (vectorized)
        flip_mask = v < 0
        u_sym = torch.where(flip_mask, -u, u)
        v_sym = torch.where(flip_mask, -v, v)

        # Grid indices calculation
        p = ((u_sym + np.pi) * N0 / (2 * np.pi)).floor().to(torch.int64)
        q = ((v_sym + np.pi) * N1 / (2 * np.pi)).floor().to(torch.int64)

        # Validity mask
        valid_mask = (p >= 0) & (p < N0) & (q >= 0) & (q < N1)
        p_valid = p[valid_mask]
        q_valid = q[valid_mask]

        if p_valid.numel() == 0:
                return (
                    torch.empty((0,), dtype=dtype, device=device),
                    valid_mask.to(device),
                )

        uvInd = (p_valid * N1 + q_valid).to(torch.int64)

        counts = torch.bincount(uvInd, minlength=N0 * N1).to(dtype)
        weights = 1.0 / torch.clamp(counts[uvInd], min=1.0)

        return weights, valid_mask
    
    @staticmethod
    def natural_weighting(
        u: torch.Tensor,
        v: torch.Tensor,
        im_size: torch.Tensor,
        weight_gridsize: int = 1,
        device=DEFAULT_DEVICE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Natural weighting: w_i = 1 for all visibilities.

        Returns:
            weights:    [N] all-ones weights for valid visibilities
            valid_mask: [N_total] boolean mask of in-bounds visibilities
        """

        dtype = torch.float32
        N0, N1 = [int(i * weight_gridsize) for i in im_size]

        flip_mask = v < 0
        u_sym = torch.where(flip_mask, -u, u)
        v_sym = torch.where(flip_mask, -v, v)

        p = ((u_sym + np.pi) * N0 / (2 * np.pi)).floor().to(torch.int64)
        q = ((v_sym + np.pi) * N1 / (2 * np.pi)).floor().to(torch.int64)

        valid_mask = (p >= 0) & (p < N0) & (q >= 0) & (q < N1)
        n_valid = int(valid_mask.sum().item())

        if n_valid == 0:
            return (
                torch.empty((0,), dtype=dtype, device=device),
                valid_mask.to(device),
            )

        weights = torch.ones(n_valid, dtype=dtype, device=device)
        return weights, valid_mask.to(device)

    @staticmethod
    def briggs_weighting(
        u: torch.Tensor,
        v: torch.Tensor,
        im_size: torch.Tensor,
        robust: float = 0.0,
        weight_gridsize: int = 1,
        device=DEFAULT_DEVICE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Briggs robust weighting.

        Args:
            robust: Briggs parameter in [-2, +2].
                    -2 ≈ uniform  (high resolution, more noise)
                     0   balanced (default)
                    +2 ≈ natural  (best sensitivity)

        Returns:
            weights:    [N] Briggs weights for valid visibilities
            valid_mask: [N_total] boolean mask of in-bounds visibilities
        """

        dtype = torch.float32
        N0, N1 = [int(i * weight_gridsize) for i in im_size]

        flip_mask = v < 0
        u_sym = torch.where(flip_mask, -u, u)
        v_sym = torch.where(flip_mask, -v, v)

        p = ((u_sym + np.pi) * N0 / (2 * np.pi)).floor().to(torch.int64)
        q = ((v_sym + np.pi) * N1 / (2 * np.pi)).floor().to(torch.int64)

        valid_mask = (p >= 0) & (p < N0) & (q >= 0) & (q < N1)
        p_valid = p[valid_mask]
        q_valid = q[valid_mask]

        if p_valid.numel() == 0:
            return (
                torch.empty((0,), dtype=dtype, device=device),
                valid_mask.to(device),
            )

        uvInd = (p_valid * N1 + q_valid).to(torch.int64)
        counts = torch.bincount(uvInd, minlength=N0 * N1).to(dtype)  # n_i per cell

        # Briggs f^2 factor (calibrated so robust=±2 matches natural/uniform)
        N_total = float(p_valid.numel())
        sum_n2  = float((counts ** 2).sum().item())
        f2 = (5.0 * 10.0 ** (-robust)) ** 2 * N_total / (sum_n2 + 1e-30)

        n_i = counts[uvInd]                              # density at each sample
        weights = 1.0 / (1.0 + n_i * f2)

        return weights.to(device), valid_mask.to(device)

    @staticmethod
    def display_uv_coverage(
        uv_normalized: torch.Tensor,
    ):
        """Display UV coverage"""

        plt.figure(figsize=(6, 6))
        plt.scatter(uv_normalized[0, :], uv_normalized[1, :], s=0.01)
        plt.xlabel("u (m)")
        plt.ylabel("v (m)")
        plt.title("Plan UV des visibilités")
        plt.grid()
        plt.axis("equal")
        plt.show()

    @staticmethod
    def bin_uv_data(
        uv_coords: torch.Tensor,
        visibilities: torch.Tensor,
        weights: torch.Tensor,
        grid_size: int = 512,
        device: torch.device = DEFAULT_DEVICE,
    ):
        """Bin UV data to reduce number of visibilities


        Args:
        device: torch.device
            Device to use for computations, it should match the self device.
        """

        """Reduce visibilities by weighted gridding onto a coarser UV grid.

        Returns:
            binned_uv:  [2, M]    weighted-mean UV coordinates per non-empty cell
            w_bin:      [M]       summed weights per cell (pass to physics.setWeight)
            vis_binned: [1, 1, M] weighted-mean visibility per cell
        """

        vis = visibilities.squeeze().to(torch.complex64)     # [N]
        w   = weights.squeeze().to(torch.float32)            # [N]
        u   = uv_coords[0].to(torch.float32)                 # [N]
        v   = uv_coords[1].to(torch.float32)                 # [N]

        # Grid indices: [-pi, pi] -> [0, grid_size)
        p = ((u + np.pi) * grid_size / (2 * np.pi)).floor().clamp(0, grid_size - 1).to(torch.int64)
        q = ((v + np.pi) * grid_size / (2 * np.pi)).floor().clamp(0, grid_size - 1).to(torch.int64)

        idx      = p * grid_size + q   # [N]
        max_bins = grid_size * grid_size

        sum_wu  = torch.zeros(max_bins, dtype=torch.float32, device=device)
        sum_wv  = torch.zeros(max_bins, dtype=torch.float32, device=device)
        sum_wvr = torch.zeros(max_bins, dtype=torch.float32, device=device)
        sum_wvi = torch.zeros(max_bins, dtype=torch.float32, device=device)
        sum_w   = torch.zeros(max_bins, dtype=torch.float32, device=device)

        sum_wu.index_add_(0, idx, w * u)
        sum_wv.index_add_(0, idx, w * v)
        sum_wvr.index_add_(0, idx, w * vis.real.to(torch.float32))
        sum_wvi.index_add_(0, idx, w * vis.imag.to(torch.float32))
        sum_w.index_add_(0, idx, w)

        mask = sum_w > 0
        w_bin      = sum_w[mask]                                          # [M]
        u_binned   = sum_wu[mask]  / w_bin                                # [M]
        v_binned   = sum_wv[mask]  / w_bin                                # [M]
        vis_binned = (sum_wvr[mask] + 1j * sum_wvi[mask]) / w_bin        # [M]

        binned_uv  = torch.stack([u_binned, v_binned], dim=0)            # [2, M]
        vis_binned = vis_binned.unsqueeze(0).unsqueeze(0)                 # [1, 1, M]

        return binned_uv, w_bin, vis_binned

    def create_deepinv_physics(
        self,
        visibility_path: Path,
        visibility_format: str,
        visibility_column: str,
        weighting: Literal["uniform", "natural", "briggs"] = "uniform",
        briggs_robust: float = 0.0,
        bin_data: bool = False,
        imaging_npixel: Optional[int] = None,
        binning_factor: Optional[float] = None,
    ):
        
        # Update default paramseters if provided
        imaging_npixel = (
            imaging_npixel if imaging_npixel is not None else self.config.imaging_npixel
        )
        binning_factor = (
            binning_factor if binning_factor is not None else self.config.binning_factor
        )

        # Load data
        uvw, visibilities, freqs = self.load_visibilities(
            visibility_path, visibility_format, visibility_column
        )
        # Normalize uv coords and compute weights
        samples_locs, visibilities = self.normalize_uv_coords(
            uvw, freqs, visibilities
        )

        im_size = torch.tensor(
            [imaging_npixel, imaging_npixel], device=self.device
        )

        if weighting == "uniform":
            weights, valid_mask = self.uniform_weighting(samples_locs[0], samples_locs[1], im_size)
        elif weighting == "natural":
            weights, valid_mask = self.natural_weighting(samples_locs[0], samples_locs[1], im_size)
        elif weighting == "briggs":
            weights, valid_mask = self.briggs_weighting(
                samples_locs[0], samples_locs[1], im_size, robust=briggs_robust
            )
        else:
            raise ValueError(f"Unknown weighting scheme '{weighting}'. Choose from: uniform, natural, briggs")
        samples_locs = samples_locs[:, valid_mask]
        visibilities = visibilities[:, :, valid_mask]

        if bin_data:
            samples_locs, weights, visibilities = self.bin_uv_data(
                samples_locs,
                visibilities,
                weights,
                grid_size=int(imaging_npixel * binning_factor),
                device=self.device,
            )

        physics = MyRadioInterferometry(
            img_size=torch.tensor([imaging_npixel, imaging_npixel]),
            samples_loc=samples_locs,
            real_projection=True,
            device=self.device,
        )

        if self.verbose:
            print("visibilities", visibilities.shape)  # [N, channels, pol]
            print("samples_locs", samples_locs.shape)  # [2, N]
            print("weights", weights.shape)  # [channels, N]

        return physics, visibilities, weights
    
    def read_ms(
            self,
        visibility_path: Path,
        visibility_format: str,
        visibility_column: str,
        weighting: Literal["uniform", "natural", "briggs"] = "uniform",
        briggs_robust: float = 0.0,
        bin_data: bool = False,
        imaging_npixel: Optional[int] = None,
        binning_factor: Optional[float] = None,
    ):
        # Update default paramseters if provided
        imaging_npixel = (
            imaging_npixel if imaging_npixel is not None else self.config.imaging_npixel
        )
        binning_factor = (
            binning_factor if binning_factor is not None else self.config.binning_factor
        )

        # Load data
        uvw, visibilities, freqs = self.load_visibilities(
            visibility_path, visibility_format, visibility_column
        )
        # Normalize uv coords and compute weights
        samples_locs, visibilities = self.normalize_uv_coords(
            uvw, freqs, visibilities
        )

        im_size = torch.tensor(
            [imaging_npixel, imaging_npixel], device=self.device
        )
        if weighting == "uniform":
            weights, valid_mask = self.uniform_weighting(samples_locs[0], samples_locs[1], im_size)
        elif weighting == "natural":
            weights, valid_mask = self.natural_weighting(samples_locs[0], samples_locs[1], im_size)
        elif weighting == "briggs":
            weights, valid_mask = self.briggs_weighting(
                samples_locs[0], samples_locs[1], im_size, robust=briggs_robust
            )
        else:
            raise ValueError(f"Unknown weighting scheme '{weighting}'. Choose from: uniform, natural, briggs")
        samples_locs = samples_locs[:, valid_mask]
        visibilities = visibilities[:, :, valid_mask]

        if bin_data:
            samples_locs, weights, visibilities = self.bin_uv_data(
                samples_locs,
                visibilities,
                weights,
                grid_size=int(imaging_npixel * binning_factor),
                device=self.device,
            )

        if self.verbose:
            print("visibilities", visibilities.shape)  # [N, channels, pol]
            print("samples_locs", samples_locs.shape)  # [2, N]
            print("weights", weights.shape)  # [channels, N]

        return samples_locs, visibilities, weights
    

    def create_psf(
        self,
        visibility_path: str,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        bin_data: bool = False,
    ):
        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )

        physics, visibilities, weights = self.create_deepinv_physics(
            visibility_path,
            visibility_format,
            visibility_column,
            bin_data=bin_data,
        )

        # Compute and normalize by PSF for calibrator
        psf = physics.A_adjoint(torch.ones_like(visibilities))

        if self.verbose:
            psf_peak = psf.max()
            print("psf_peak", psf_peak)

        return psf

    def create_dirty_image(
        self,
        visibility_path: str,
        visibility_format: str = "MS",
        visibility_column: str = "DATA",
        bin_data: bool = False,
    ):
        if visibility_format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility_format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )

        physics, visibilities, weights = self.create_deepinv_physics(
            visibility_path,
            visibility_format,
            visibility_column,
            bin_data=bin_data,
        )

        back = physics.A_adjoint(visibilities)

        # Compute and normalize by PSF for calibrator
        psf = physics.A_adjoint(torch.ones_like(visibilities))
        psf_peak = psf.max()
        if self.verbose:
            print("psf_peak", psf_peak)

        back_normalized = back / psf_peak

        if self.verbose:
            print("Backprojection min value: ", back_normalized.min().item())
            print("Backprojection peak value: ", back_normalized.max().item())

        return back_normalized

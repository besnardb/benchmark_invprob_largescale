import hashlib
import yaml
import json
import types
import math
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    AltAz,
    ICRS,
)
from astropy.time import Time
import astropy.units as u
from scipy.ndimage import zoom

MEERKAT_LOCATION = EarthLocation(
    lat=-30.83 * u.deg, lon=21.33 * u.deg, height=1195.0 * u.m
)

SKALOW_LOCATION = EarthLocation(
    lat=-26.7 * u.deg,
    lon=116.76 * u.deg,
    height=300.0 * u.m
)

LOFAR_LOCATION = EarthLocation(
    lat=52.915 * u.deg,
    lon=6.869 * u.deg,
    height=50.0 * u.m
)

def load_and_resize_image(image_path, image_size, normalize=False):
    """Load and resize a FITS image.

    Returns
    -------
    np.ndarray
        The resized image of shape (C, H, W) in [0, 1].
    """
    with fits.open(image_path, memmap=False) as hdul:
        img = np.array(hdul[0].data, dtype=np.float32, copy=True)

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(img))

    # Normalize to [0, 1]
    if normalize and max_val > 1.0:
        img = img / max_val

    img = np.squeeze(img)

    # Ensure (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim != 3:
        raise ValueError(
            f"Unexpected FITS image shape {img.shape}, expected 2D or 3D after squeeze."
        )

    c, h, w = img.shape
    resized_img = img.copy()

    if h != image_size or w != image_size:
        zoom_factors = (1, image_size / h, image_size / w)
        resized_img = zoom(img, zoom_factors, order=3)
        #resized_img = np.clip(resized_img, 0, 1)

    return np.ascontiguousarray(resized_img, dtype=np.float32)


def load_new_header(fits_file, image_size):
    orig_header = fits.getheader(fits_file)
    orig_naxis1 = int(orig_header.get("NAXIS1", image_size))
    orig_naxis2 = int(orig_header.get("NAXIS2", image_size))
    new_header = orig_header.copy()
    # Scale pixel size: new pixel covers more angle (fewer pixels, same FOV)
    if "CDELT1" in new_header:
        new_header["CDELT1"] = float(orig_header["CDELT1"]) * orig_naxis1 / image_size
    if "CDELT2" in new_header:
        new_header["CDELT2"] = float(orig_header["CDELT2"]) * orig_naxis2 / image_size
    # Move reference pixel to the centre of the new image
    new_header["CRPIX1"] = (image_size + 1) / 2.0
    new_header["CRPIX2"] = (image_size + 1) / 2.0
    return new_header


'''def get_meerkat_visibilities_path(
    image: np.ndarray,
    cache_dir: Path,
    start_frequency_hz: float = 1e9,
    number_of_time_steps: int = 256,
    integral_time: float = 10, # 10 sec integration
):
    """
    Generate path for MeerKAT visibilities.
    """
    # Create a unique hash for the simulation parameters
    params = {
        'start_frequency_hz': start_frequency_hz,
        'number_of_time_steps': number_of_time_steps,
        'integral_time': integral_time
    }
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()

    if hasattr(image, "cpu") and hasattr(image, "numpy"):
        img_bytes = image.cpu().numpy().tobytes()
    else:
        img_bytes = image.tobytes()

    img_hash = hashlib.md5(img_bytes).hexdigest()
    full_hash = hashlib.md5((params_hash + img_hash).encode()).hexdigest()

    vis_path = cache_dir / f"{full_hash}.ms"
    return vis_path'''


def get_meerkat_visibilities_path(
    image: np.ndarray,
    cache_dir: Path,
    fits_name: str,
    imaging_npixel: int,
    number_of_time_steps: int = 256,
    start_frequency_hz: float = 100e6,
    end_frequency_hz: float = 120e6,
    number_of_channels: int = 12,
    random_position: bool = False,
):
    """
    Generate path for MeerKAT visibilities.
    """
    # Create a unique hash for the simulation parameters
    params = {
        "fits_name": fits_name,
        "number_of_time_steps": number_of_time_steps,
        "start_frequency_hz": start_frequency_hz,
        "end_frequency_hz": end_frequency_hz,
        "number_of_channels": number_of_channels,
        "random_position": random_position,
        "imaging_npixel": imaging_npixel,
    }
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()

    if hasattr(image, "cpu") and hasattr(image, "numpy"):
        img_bytes = np.ascontiguousarray(
            image.cpu().numpy(), dtype=np.float32
        ).tobytes()
    else:
        img_bytes = np.ascontiguousarray(image, dtype=np.float32).tobytes()

    img_hash = hashlib.md5(img_bytes).hexdigest()
    full_hash = hashlib.md5((params_hash + img_hash).encode()).hexdigest()

    vis_path = cache_dir / f"{full_hash}.ms"
    return vis_path


def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_config(config_path, section=None):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    if section is not None:
        if not hasattr(cfg, section):
            raise KeyError(f"Section '{section}' not found in config: {config_path}")
        return getattr(cfg, section)

    return cfg


def is_source_visible(
    ra_deg,
    dec_deg,
    obs_start_time,
    obs_duration,
    telescope_location,
    min_elevation_deg=15.0,
    n_time_samples=10,
):
    """
    Check if a source is visible (above horizon) for the entire observation duration.

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        obs_start_time: Observation start time (datetime object)
        obs_duration: Observation duration (timedelta object)
        telescope_location: EarthLocation of the telescope
        min_elevation_deg: Minimum elevation above horizon in degrees (default: 15)
        n_time_samples: Number of time samples to check during observation

    Returns:
        bool: True if source is visible for entire observation, False otherwise
    """
    # Create sky coordinate
    source = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    # Sample times throughout the observation
    time_samples = [
        obs_start_time + i * obs_duration / (n_time_samples - 1)
        for i in range(n_time_samples)
    ]

    # Check elevation at each time sample
    for t in time_samples:
        obs_time = Time(t)
        altaz_frame = AltAz(obstime=obs_time, location=telescope_location)
        source_altaz = source.transform_to(altaz_frame)

        if source_altaz.alt.deg < min_elevation_deg:
            return False

    return True


def draw_random_pointing(
    time: Time,
    observer: EarthLocation = MEERKAT_LOCATION,
    min_elevation_deg: float = 30.0,
    max_attempts: int = 1000,
    n_azimuth_samples: int = 360,
) -> tuple:
    """Draw a random pointing (RA/Dec) inside the region defined by a minimum elevation contour.

    This function computes the elevation contour at a given time and observer location,
    then randomly samples a point within that region.

    Args:
        time: Time at which to draw the sample
        observer: Earth location of the observer
        min_elevation_deg: Minimum elevation in degrees defining the contour
        max_attempts: Maximum number of random attempts to find a valid point
        n_azimuth_samples: Number of azimuth samples to define the contour boundary

    Returns:
        tuple: (ra_deg, dec_deg) coordinates of the random pointing in degrees

    Raises:
        RuntimeError: If no valid pointing is found after max_attempts

    Example:
        >>> from astropy.time import Time
        >>> from astropy.coordinates import EarthLocation
        >>> import astropy.units as u
        >>>
        >>> time = Time("2020-04-26T16:36:00")
        >>> meerkat = EarthLocation(lat=-30.83*u.deg, lon=21.33*u.deg, height=1195.0*u.m)
        >>> ra, dec = draw_random_pointing_in_elevation_contour(
        ...     time=time,
        ...     observer=meerkat,
        ...     min_elevation_deg=15.0
        ... )
    """
    # Sample azimuth angles to define the elevation contour boundary
    azimuth_sample = np.linspace(0, 360, n_azimuth_samples)
    elevation_boundary = np.full(azimuth_sample.size, min_elevation_deg)

    # Convert boundary to RA/Dec
    boundary_altaz = SkyCoord(
        azimuth_sample * u.deg,
        elevation_boundary * u.deg,
        frame=AltAz(obstime=time, location=observer),
    )
    boundary_radec = boundary_altaz.transform_to(ICRS)

    # Get RA/Dec ranges from the boundary
    ra_deg = boundary_radec.ra.deg
    dec_deg = boundary_radec.dec.deg

    # Compute approximate bounds (accounting for RA wrapping)
    dec_min = np.min(dec_deg)
    dec_max = np.max(dec_deg)

    # For RA, handle potential wrapping around 0/360
    ra_range = np.ptp(ra_deg)  # peak-to-peak (max - min)
    if ra_range > 180:  # Wrapping detected
        # Use the full RA range
        ra_min = 0
        ra_max = 360
    else:
        ra_min = np.min(ra_deg)
        ra_max = np.max(ra_deg)

    # Randomly sample points within the bounding box
    for attempt in range(max_attempts):
        # Generate random RA/Dec within bounds
        ra_candidate = ra_min + np.random.rand() * (ra_max - ra_min)
        dec_candidate = dec_min + np.random.rand() * (dec_max - dec_min)

        # Convert to AltAz to check elevation
        candidate_radec = SkyCoord(
            ra_candidate * u.deg, dec_candidate * u.deg, frame=ICRS
        )
        candidate_altaz = candidate_radec.transform_to(
            AltAz(obstime=time, location=observer)
        )

        # Check if elevation meets the minimum requirement
        if candidate_altaz.alt.deg >= min_elevation_deg:
            return ra_candidate, dec_candidate

    raise RuntimeError(
        f"Could not find valid pointing within elevation contour after {max_attempts} attempts. "
        f"Try increasing max_attempts or reducing min_elevation_deg."
    )


def get_cellsize_from_fits_wcs(fits_file: Path) -> float:
    """Return pixel angular size (radians/pixel) from FITS WCS."""
    header = fits.getheader(fits_file)
    cdelt1 = header.get("CDELT1")
    cdelt2 = header.get("CDELT2")

    if cdelt1 is None and cdelt2 is None:
        raise ValueError("FITS header has no CDELT1/CDELT2")

    if cdelt1 is not None and cdelt2 is not None:
        # Use both axes when available for robustness to tiny anisotropy.
        pixel_scale_deg = 0.5 * (abs(float(cdelt1)) + abs(float(cdelt2)))
    else:
        pixel_scale_deg = abs(float(cdelt1 if cdelt1 is not None else cdelt2))

    return math.radians(pixel_scale_deg)

def meerkat_noise_rms(freq_hz: float,
                      bandwidth_hz: float,
                      integration_time_s: float,
                      eta_s: float = 1.0) -> float:
    
    k_B = 1.380649e-23  # Boltzmann constant
    D = 13.5            # Dish diameter [m]
    sigma_surf = 600e-6  # Surface RMS [m]
    
    if freq_hz < 0.58e9 or freq_hz > 3.05e9:
        raise ValueError("Frequency out of MeerKAT operational range")
    
    freq_GHz = freq_hz / 1e9
    lam = 3e8 / freq_hz
    
    if freq_hz >= 0.58e9 and freq_hz < 1.02e9: # Band UHF
        T_rcv = 11 - 4.5 * (freq_GHz - 0.58)
    elif freq_hz >= 1.02e9 and freq_hz < 1.67e9: # Band L
        T_rcv = 7.5 + 6.8 * ((np.abs(freq_GHz - 1.65))**1.5)
    elif freq_hz >= 1.67e9 and freq_hz <= 3.05e9: # Band S
        T_rcv = 7.5

    T_sys = T_rcv + 2.73 + 4  # T_sky + T_spl

    # Efficiencies
    hF = 0.80 - 0.04 * np.abs(np.log10(freq_GHz))
    hD = 1 - 20 * (lam / D)**1.5
    eps_phase = np.exp(-(4 * np.pi * sigma_surf / lam)**2)
    eta_ap = hF * hD * eps_phase

    A_eff = eta_ap * np.pi * (D/2)**2

    # SEFD en Jy
    sefd_jy = (2 * k_B * T_sys) / A_eff * 1e26

    sigma_noise = sefd_jy / (eta_s * np.sqrt(2 * bandwidth_hz * integration_time_s))

    return sigma_noise

def lofar_sefd_jy(freq_hz: float, mode: str = "HBA") -> float:

    freq_mhz = freq_hz / 1e6

    if mode.upper() == "LBA":

        if not 10 <= freq_mhz <= 90:
            raise ValueError("LBA frequency must be between 10 and 90 MHz.")

        # Simple sky-dominated approximation.
        sefd = 40e3 * (freq_mhz / 60.0) ** (-2.5)

        sefd = max(sefd, 40e3)

        if freq_mhz > 70:
            sefd *= 1.0 + 1.0 * (freq_mhz - 70.0) / 20.0

        return sefd

    elif mode.upper() == "HBA":
        if not 110 <= freq_mhz <= 240:
            raise ValueError("HBA frequency must be between 110 and 240 MHz.")

        # Approximate interpolation of LOFAR HBA Core station SEFD.
        freq_table = np.array([
            120, 150, 180, 200, 210, 240
        ])

        sefd_table = np.array([
            2.5e3,
            2.8e3,
            3.3e3,
            3.8e3,
            4.2e3,
            5.0e3,
        ])

        return float(np.interp(freq_mhz, freq_table, sefd_table))

    else:
        raise ValueError("mode must be 'LBA' or 'HBA'.")


def lofar_noise_rms(
    freq_hz: float,
    bandwidth_hz: float,
    integration_time_s: float,
    mode: str = "HBA",
) -> float:

    sefd_jy = lofar_sefd_jy(freq_hz, mode)

    return sefd_jy / np.sqrt(
        2.0 * bandwidth_hz * integration_time_s
    )
from scipy.signal import windows, savgol_filter
import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube

def senti2bap(scl, data_cube, reducer="first"):
    """
    Apply cloud and shadow masking to Sentinel-2 data using Scene Classification Layer (SCL).

    Parameters:
    scl (array-like): Scene Classification Layer band.
    data_cube: Sentinel-2 data cube containing spectral bands.
    reducer (str): Reduction method to apply (default: "first").

    Returns:
    Masked data cube with selected pixels.
    """
    # Mask clouds and shadows (valid SCL values)
    mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)

    # Apply Gaussian smoothing to the mask
    g = windows.gaussian(11, std=1.6)
    kernel = np.outer(g, g)
    kernel = kernel / kernel.sum()

    # Morphological dilation of mask: convolution + thresholding
    mask = mask.apply_kernel(kernel)
    mask = mask > 0.1

    # Apply the mask to the entire data cube
    masked_data = data_cube.mask(mask)

    return masked_data.reduce_dimension(reducer=reducer, dimension="t")


def compute_sentinel2_indices(sentinel2_data_cube, indices=None, apply_mask=False):
    """
    Compute selected Sentinel-2 vegetation indices and band combinations.

    Parameters:
    sentinel2_data_cube: Sentinel-2 data cube object where bands are accessed using `band()`.
    indices (list of str, optional): List of indices to compute. If None, all indices will be computed.
    apply_mask (bool, optional): Whether to apply cloud and shadow masking.

    Returns:
    dict: A dictionary containing the requested vegetation indices.
    """

    # Load Sentinel-2 bands
    bands = {
        "coastal": sentinel2_data_cube.band("B01"),
        "blue": sentinel2_data_cube.band("B02"),
        "green": sentinel2_data_cube.band("B03"),
        "red": sentinel2_data_cube.band("B04"),
        "nir": sentinel2_data_cube.band("B08"),
        "nir_08": sentinel2_data_cube.band("B8A"),
        "nir_09": sentinel2_data_cube.band("B09"),
        "swir_16": sentinel2_data_cube.band("B11"),
        "swir_22": sentinel2_data_cube.band("B12"),
        "scl": sentinel2_data_cube.band("SCL"),
    }

    # Apply cloud and shadow masking
    if apply_mask:
        sentinel2_data_cube = senti2bap(bands["scl"], sentinel2_data_cube, reducer="first")
        bands = {k: sentinel2_data_cube.band(k) for k in bands.keys()}  # Reload masked bands

    # Define index formulas
    
    index_calculations = {
        "true_color": lambda: (bands["red"], bands["green"], bands["blue"]),
        "color_infrared": lambda: (bands["nir"], bands["red"], bands["green"]),
        "shortwave_infrared": lambda: (bands["swir_22"], bands["nir_08"], bands["red"]),
        "agriculture": lambda: (bands["swir_16"], bands["nir"], bands["blue"]),
        "ndvi": lambda: (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"]),
        "gndvi": lambda: (bands["nir"] - bands["green"]) / (bands["nir"] + bands["green"]),
        "evi": lambda: 2.5 * ((bands["nir"] - bands["red"]) / (bands["nir"] + 6 * bands["red"] - 7.5 * bands["blue"] + 1)),
        "avi": lambda: bands["nir"] * (1 - bands["red"]) * (bands["nir"] - bands["red"]),
        "savi": lambda: ((bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"] + 0.428)) * 1.428,
        "ndmi": lambda: (bands["nir"] - bands["swir_16"]) / (bands["nir"] + bands["swir_16"]),
        "msi": lambda: bands["swir_16"] / bands["nir"],
        "gci": lambda: bands["nir"] / bands["green"],
        "nbri": lambda: (bands["nir"] - bands["swir_22"]) / (bands["nir"] + bands["swir_22"]),
        "bsi": lambda: ((bands["swir_16"] + bands["red"]) - (bands["nir"] + bands["blue"])) / ((bands["swir_16"] + bands["red"]) + (bands["nir"] + bands["blue"])),
        "ndwi": lambda: (bands["green"] - bands["nir"]) / (bands["green"] + bands["nir"]),
        "ndsi": lambda: (bands["green"] - bands["swir_16"]) / (bands["green"] + bands["swir_16"]),
        "ndgi": lambda: (bands["green"] - bands["nir"]) / (bands["green"] + bands["nir"]),
        "rvi": lambda: (bands["nir"] - (2 * bands["red"]) + bands["blue"]) / (bands["nir"] + (2 * bands["red"]) + bands["blue"]),
        "sipi": lambda: (bands["nir"] - bands["blue"]) / (bands["nir"] - bands["red"]),
        "moisture": lambda: (bands["nir_08"] - bands["swir_16"]) / (bands["nir_08"] + bands["swir_16"]),
    }

    # Compute only requested indices (or all if None)
    if indices is None:
        indices = list(index_calculations.keys())  # Compute all if not specified

    results = {index: index_calculations[index]() for index in indices if index in index_calculations}

    return results


def apply_savgol_smoothing(cube: XarrayDataCube, polyorder=2):
    """
    Apply Savitzky-Golay smoothing to an OpenEO XarrayDataCube.

    Parameters:
    cube (XarrayDataCube): Input data cube with a time dimension ('t').
    polyorder (int): Polynomial order for the Savitzky-Golay filter (default: 2).

    Returns:
    XarrayDataCube: Smoothed data cube.
    """
    array = cube.get_array()
    time_steps = array.sizes["t"]

    # If there are not enough time steps, return filled zeros or the original cube
    if time_steps < 2:
        return cube  
    if time_steps < 3:
        return XarrayDataCube(array.fillna(0))  

    # Forward-fill and backward-fill missing values
    filled = array.ffill(dim='t').bfill(dim='t')

    # Set window size (odd number, max 5)
    window_size = min(time_steps, 5)
    if window_size % 2 == 0:
        window_size -= 1  

    # Apply Savitzky-Golay smoothing
    smoothed_array = savgol_filter(filled.values, window_length=window_size, polyorder=polyorder, axis=0)

    return XarrayDataCube(xr.DataArray(smoothed_array, dims=array.dims, coords=array.coords))


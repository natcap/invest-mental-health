import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import os


def calculate_pd_layer(
        ndvi_raster_path: str,
        pop_raster_path: str,
        rr: float,
        NE_goal: float,
        baseline_risk_rate: np.ndarray,
        output_dir: str = "."
) -> dict:
    """
    Calculate Preventable Disease Impact (PD_i) raster from environmental and population data.

    Computes the potential health benefit of achieving NDVI targets through:
    1. NDVI deficit calculation (delta_NE_i)
    2. Risk reduction estimation (RR_i)
    3. Preventable fraction calculation (PF_i)
    4. Population-weighted impact (PD_i)

    Parameters:
        ndvi_raster_path: Path to NDVI raster (geotiff)
        pop_raster_path: Path to population density raster (geotiff)
        rr: Risk ratio (RR_0.1NE) from health studies
        NE_goal: Target NDVI value for health benefit
        baseline_risk_rate: Baseline disease prevalence raster (must match NDVI dimensions)
        output_dir: Output directory path (default: current directory)

    Returns:
        Dictionary containing:
        - PD_i: Preventable Disease Impact raster (numpy array)
        - PF_i: Preventable Fraction raster
        - RR_i: Risk Ratio raster
        - delta_NE_i: NDVI deficit raster
        - output_paths: Dictionary of saved raster paths

    Raises:
        ValueError: If input rasters have incompatible dimensions
        RuntimeError: If raster processing fails
    """
    # -----------------------------
    # 1. NDVI Data Preparation
    # -----------------------------
    try:
        with rasterio.open(ndvi_raster_path) as src:
            ndvi_data = src.read(1).astype(np.float32)
            ndvi_meta = src.meta.copy()
            ndvi_meta.update(dtype="float32", nodata=np.nan)

        # Mask invalid NDVI values (<0 typically indicates no data)
        ndvi_data[ndvi_data < 0] = np.nan

        # Validate baseline risk dimensions
        if baseline_risk_rate.shape != ndvi_data.shape:
            raise ValueError(
                f"Baseline risk shape {baseline_risk_rate.shape} "
                f"doesn't match NDVI {ndvi_data.shape}"
            )
    except Exception as e:
        raise RuntimeError(f"NDVI processing failed: {str(e)}")

    # -----------------------------
    # 2. NDVI Deficit Calculation
    # -----------------------------
    delta_NE_i = NE_goal - ndvi_data

    # -----------------------------
    # 3. Risk Reduction Modeling
    # -----------------------------
    RR_i = np.exp(np.log(rr) * 10 * delta_NE_i)  # Convert RR per 0.1 NDVI to actual delta
    PF_i = 1 - RR_i  # Preventable fraction

    # -----------------------------
    # 4. Population Data Processing
    # -----------------------------
    try:
        with rasterio.open(pop_raster_path) as src:
            population_data = src.read(1).astype(np.float32)
            pop_meta = src.meta.copy()

        # Resample population if needed (using bilinear interpolation)
        if population_data.shape != ndvi_data.shape:
            population_resampled = np.empty_like(ndvi_data, dtype=np.float32)
            reproject(
                source=population_data,
                destination=population_resampled,
                src_transform=pop_meta["transform"],
                src_crs=pop_meta["crs"],
                dst_transform=ndvi_meta["transform"],
                dst_crs=ndvi_meta["crs"],
                resampling=Resampling.bilinear,
            )
            population_data = population_resampled

        # Mask non-populated areas
        population_data[population_data <= 0] = np.nan
    except Exception as e:
        raise RuntimeError(f"Population data processing failed: {str(e)}")

    # -----------------------------
    # 5. Health Impact Calculation
    # -----------------------------
    PD_i = PF_i * baseline_risk_rate * population_data

    # -----------------------------
    # 6. Output Generation
    # -----------------------------
    output_paths = {}
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Save intermediate rasters
        for name, data in [
            ("ndvi_masked", ndvi_data),
            ("delta_NE_i", delta_NE_i),
            ("PD_i", PD_i)
        ]:
            path = os.path.join(output_dir, f"{name}.tif")
            with rasterio.open(path, "w", **ndvi_meta) as dst:
                dst.write(data, 1)
            output_paths[name] = path
            print(f"Saved {name} raster: {path}")

    except Exception as e:
        raise RuntimeError(f"Output saving failed: {str(e)}")

    return {
        "PD_i": PD_i,
        "PF_i": PF_i,
        "RR_i": RR_i,
        "delta_NE_i": delta_NE_i,
        "output_paths": output_paths
    }
import pandas as pd
import numpy as np
import geopandas as gpd


def load_baseline_risk_from_shapefile(
        shapefile_path: str,
        geoid_col: str = "GEOID",
        risk_col: str = "DEPRESS",
        default_risk: float = 0.15,
        target_crs: str = "EPSG:5070"
) -> gpd.GeoDataFrame:
    """
    Load baseline risk data from a shapefile with geographic attributes.

    Args:
        shapefile_path: Path to the shapefile containing risk data
        geoid_col: Column name for geographic identifiers (default: "GEOID")
        risk_col: Column name containing risk values (default: "DEPRESS")
        default_risk: Default value to fill missing risk values (default: 0.15)
        target_crs: Target coordinate reference system (default: "EPSG:5070")

    Returns:
        GeoDataFrame containing processed GEOID and baseline risk data

    Raises:
        ValueError: If required columns are missing in the shapefile
    """
    # Read shapefile into GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Validate required columns
    if geoid_col not in gdf.columns:
        raise ValueError(f"Shapefile missing required column: {geoid_col}")
    if risk_col not in gdf.columns:
        raise ValueError(f"Shapefile missing required column: {risk_col}")

    # Standardize coordinate reference system
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Ensure GEOID is string type for consistent merging
    gdf[geoid_col] = gdf[geoid_col].astype(str)

    # Fill missing risk values with default
    gdf[risk_col] = gdf[risk_col].fillna(default_risk)

    return gdf[[geoid_col, risk_col]]


def load_health_effects(
        excel_file: str,
        health_indicator_i: str,
        baseline_risk=None,
        NE_goal: float = 0.3,
        aoi_gdf: gpd.GeoDataFrame = None,
        baseline_risk_gdf: gpd.GeoDataFrame = None,
        risk_col: str = "DEPRESS"
) -> dict:
    """
    Calculate health effects from exposure data, supporting both uniform and geographic risk inputs.

    Parameters:
        excel_file: Path to Excel file containing health effect sizes
        health_indicator_i: Target health indicator (e.g., 'depression')
        baseline_risk: Optional uniform risk value (default: None)
        NE_goal: Target NDVI exposure value (default: 0.3)
        aoi_gdf: GeoDataFrame of study area (required for geographic risk)
        baseline_risk_gdf: GeoDataFrame containing geographic risk values
        risk_col: Column name for risk values in baseline_risk_gdf (default: "DEPRESS")

    Returns:
        Dictionary containing:
        - effect_size_i: Extracted effect size
        - effect_indicator_i: Effect metric type
        - risk_ratio: Calculated risk ratio(s)
        - NE_goal: Target NDVI value
        - baseline_risk: Risk value(s) used
        - aoi_gdf: Study area geometry

    Raises:
        ValueError: For invalid inputs or inconsistent health effect data
    """
    # Load and filter health effect data
    es = pd.read_excel(excel_file, sheet_name="Sheet1", usecols="A:D")
    es_selected = es[es["health_indicator"] == health_indicator_i]

    # Validate health effect data consistency
    effect_size_i = _validate_effect_size(es_selected)
    effect_indicator_i = _validate_effect_indicator(es_selected)

    # Process risk inputs (priority to geographic data)
    if baseline_risk_gdf is not None:
        if aoi_gdf is None:
            raise ValueError("aoi_gdf required for geographic risk matching")
        risk_values = _match_risk_values(aoi_gdf, baseline_risk_gdf, risk_col)
        risk_ratio = _calculate_risk_ratio(effect_size_i, effect_indicator_i, risk_values)
    else:
        baseline_risk = 0.15 if baseline_risk is None else baseline_risk
        risk_ratio = _calculate_risk_ratio(effect_size_i, effect_indicator_i, baseline_risk)

    return {
        "effect_size_i": effect_size_i,
        "effect_indicator_i": effect_indicator_i,
        "risk_ratio": risk_ratio,
        "NE_goal": NE_goal,
        "baseline_risk": baseline_risk if baseline_risk_gdf is None else risk_values,
        "aoi_gdf": aoi_gdf
    }


# --------------------------
# Internal helper functions
# --------------------------
def _validate_effect_size(es_data: pd.DataFrame) -> float:
    """Validate and extract unique effect size value."""
    values = np.unique(es_data["effect_size"].values)
    if len(values) == 1:
        return values[0]
    raise ValueError("Multiple effect_size values found - specify exact indicator")


def _validate_effect_indicator(es_data: pd.DataFrame) -> str:
    """Validate and extract unique effect indicator type."""
    indicators = np.unique(es_data["effect_indicator"].values)
    if len(indicators) == 1:
        return indicators[0]
    raise ValueError("Multiple effect_indicators found - specify exact metric")


def _match_risk_values(
        aoi_gdf: gpd.GeoDataFrame,
        risk_gdf: gpd.GeoDataFrame,
        risk_col: str
) -> np.ndarray:
    """Match risk values to study areas using GEOID."""
    aoi_gdf["GEOID"] = aoi_gdf["GEOID"].astype(str)
    merged = aoi_gdf.merge(risk_gdf, on="GEOID", how="left")
    return merged[risk_col].fillna(0.15).values  # Fill unmatched areas with default


def _calculate_risk_ratio(
        effect_size: float,
        effect_indicator: str,
        baseline_risk
) -> float:
    """
    Calculate risk ratio(s) based on effect metric type.

    Supports both scalar and array inputs for baseline_risk.
    """
    if effect_indicator == "risk ratio":
        return effect_size if isinstance(baseline_risk, float) else np.full_like(baseline_risk, effect_size)

    if effect_indicator == "odd ratio":
        if isinstance(baseline_risk, float):
            return effect_size / (1 - baseline_risk + baseline_risk * effect_size)
        return effect_size / (1 - baseline_risk + baseline_risk * effect_size)

    raise ValueError("effect_indicator must be either 'risk ratio' or 'odd ratio'")
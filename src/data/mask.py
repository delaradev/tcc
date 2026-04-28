#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EXTRACT ANA MASKS FOR VALIDATION
================================================================================

Extract center pivot masks from ANA data for a specific municipality,
convert to raster format compatible with model validation.

Input: ANA shapefiles (from data/raw/ana/)
Output: Raster masks aligned with Landsat tiles

Author: Emanuel de Lara Ruas
Version: 1.0.0
================================================================================
"""

import geopandas as gpd
import rasterio
import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ANAMaskExtractor:
    """
    Extract ANA center pivot masks for validation
    """

    # ANA data columns mapping (based on official documentation)
    ANA_COLUMNS = {
        'id': 'ID_PIVO',           # Pivot identifier
        'area': 'AREA_HA',         # Area in hectares
        'state': 'UF',             # State (RS, MG, etc)
        'municipality': 'MUNICIPIO',  # Municipality name
        'lat': 'LATITUDE',         # Center latitude
        'lon': 'LONGITUDE',        # Center longitude
        'radius': 'RAIO_M',        # Radius in meters
    }

    def __init__(
        self,
        ana_path: Path,
        target_municipality: str = "Salto do Jacuí",
        target_state: str = "RS",
        target_year: int = 2023,
        buffer_meters: int = 100
    ):
        """
        Initialize extractor

        Args:
            ana_path: Path to ANA data directory
            target_municipality: Municipality name
            target_state: State abbreviation
            target_year: Year of data
            buffer_meters: Buffer around pivots (meters)
        """
        self.ana_path = Path(ana_path)
        self.target_municipality = target_municipality
        self.target_state = target_state
        self.target_year = target_year
        self.buffer_meters = buffer_meters

        self.ana_data: Optional[gpd.GeoDataFrame] = None
        self.filtered_data: Optional[gpd.GeoDataFrame] = None
        self.mask: Optional[np.ndarray] = None

    def load_ana_data(self, shapefile: Optional[Path] = None) -> gpd.GeoDataFrame:
        """
        Load ANA shapefile

        Args:
            shapefile: Specific shapefile path (auto-detect if None)

        Returns:
            GeoDataFrame with ANA data
        """
        if shapefile is None:
            # Auto-detect shapefile for target state and year
            shapefile_pattern = f"*{self.target_state}*{self.target_year}*.shp"
            shapefiles = list(self.ana_path.glob(shapefile_pattern))

            if not shapefiles:
                # Try more general pattern
                shapefiles = list(self.ana_path.glob(
                    f"*{self.target_state}*.shp"))

            if not shapefiles:
                raise FileNotFoundError(
                    f"No shapefile found for {self.target_state} in {self.ana_path}"
                )

            shapefile = shapefiles[0]
            logger.info(f"Found shapefile: {shapefile.name}")

        logger.info(f"Loading ANA data from: {shapefile}")
        self.ana_data = gpd.read_file(shapefile)

        # Filter by state if column exists
        if self.ANA_COLUMNS['state'] in self.ana_data.columns:
            self.ana_data = self.ana_data[
                self.ana_data[self.ANA_COLUMNS['state']] == self.target_state
            ]
            logger.info(
                f"Filtered by state {self.target_state}: {len(self.ana_data)} pivots")

        logger.info(f"Loaded {len(self.ana_data)} pivots")
        return self.ana_data

    def filter_by_municipality(self) -> gpd.GeoDataFrame:
        """
        Filter ANA data by municipality name

        Returns:
            Filtered GeoDataFrame
        """
        if self.ana_data is None:
            raise ValueError("Load ANA data first with load_ana_data()")

        # Try different column name variations
        possible_columns = ['MUNICIPIO', 'NM_MUN', 'MUN', 'MUNICIPIO_NM']

        mun_col = None
        for col in possible_columns:
            if col in self.ana_data.columns:
                mun_col = col
                break

        if mun_col is None:
            logger.warning(
                f"Municipality column not found. Available: {self.ana_data.columns.tolist()}")
            return self.ana_data

        # Filter with flexible matching
        self.filtered_data = self.ana_data[
            self.ana_data[mun_col].str.contains(
                self.target_municipality,
                case=False,
                na=False
            )
        ]

        logger.info(
            f"Found {len(self.filtered_data)} pivots in {self.target_municipality}")

        if len(self.filtered_data) == 0:
            logger.warning(f"No pivots found in {self.target_municipality}")

            # List available municipalities in state
            municipalities = self.ana_data[mun_col].unique()[:20]
            logger.info(
                f"First 20 municipalities in {self.target_state}: {municipalities}")

        return self.filtered_data

    def create_pivot_mask(
        self,
        reference_raster: Optional[Path] = None,
        output_resolution: float = 30.0,
        output_crs: str = 'EPSG:31982',  # SIRGAS 2000 / UTM zone 22S (RS)
        output_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> np.ndarray:
        """
        Create binary mask from ANA pivot geometries

        Args:
            reference_raster: Path to reference raster (to get bounds and CRS)
            output_resolution: Resolution in meters (if no reference)
            output_crs: CRS for output mask
            output_bounds: (minx, miny, maxx, maxy) bounds

        Returns:
            Binary mask array (1 = pivot, 0 = background)
        """
        if self.filtered_data is None or len(self.filtered_data) == 0:
            raise ValueError(
                "No filtered data. Run filter_by_municipality() first.")

        # Get geometries (create circles if not already polygons)
        geometries = []

        for idx, row in self.filtered_data.iterrows():
            geom = row.geometry

            # If geometry is point (center), create circle
            if geom.geom_type == 'Point':
                # Default 100m
                radius = row.get(self.ANA_COLUMNS['radius'], 100)
                circle = geom.buffer(radius)
                geometries.append(circle)
            else:
                geometries.append(geom)

        # Union all geometries
        union_geom = unary_union(geometries)

        # Determine bounds
        if reference_raster and reference_raster.exists():
            # Get bounds from reference raster
            with rasterio.open(reference_raster) as src:
                bounds = src.bounds
                transform = src.transform
                output_crs = src.crs
                width = src.width
                height = src.height

            logger.info(f"Using reference raster bounds: {bounds}")

            # Burn geometries to raster
            mask = features.rasterize(
                [(union_geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

        elif output_bounds:
            # Create new raster from bounds
            minx, miny, maxx, maxy = output_bounds
            width = int((maxx - minx) / output_resolution)
            height = int((maxy - miny) / output_resolution)
            transform = from_origin(
                minx, maxy, output_resolution, output_resolution)

            mask = features.rasterize(
                [(union_geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

        else:
            # Auto-detect bounds from geometries
            bounds = union_geom.bounds
            width = int((bounds[2] - bounds[0]) / output_resolution)
            height = int((bounds[3] - bounds[1]) / output_resolution)
            transform = from_origin(
                bounds[0], bounds[3], output_resolution, output_resolution)

            mask = features.rasterize(
                [(union_geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

        self.mask = mask
        self.mask_transform = transform
        self.mask_crs = output_crs

        logger.info(
            f"Created mask: {mask.shape} pixels, {mask.sum()} pivot pixels")

        return mask

    def save_mask(
        self,
        output_path: Path,
        transform: Optional[rasterio.Affine] = None,
        crs: Optional[str] = None
    ) -> Path:
        """
        Save mask as GeoTIFF

        Args:
            output_path: Output file path
            transform: Raster transform (uses stored if None)
            crs: Coordinate reference system (uses stored if None)

        Returns:
            Output path
        """
        if self.mask is None:
            raise ValueError("No mask created. Run create_pivot_mask() first.")

        transform = transform or self.mask_transform
        crs = crs or self.mask_crs

        if transform is None or crs is None:
            raise ValueError("Missing transform or CRS. Provide explicitly.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=self.mask.shape[0],
            width=self.mask.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(self.mask, 1)

        logger.info(f"Saved mask to: {output_path}")

        return output_path

    def compare_with_predictions(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5
    ) -> dict:
        """
        Compare ANA mask with model predictions

        Args:
            predictions: Model prediction array (same shape as mask)
            threshold: Classification threshold

        Returns:
            Dictionary with IoU, Precision, Recall, F1
        """
        if self.mask is None:
            raise ValueError("No mask created. Run create_pivot_mask() first.")

        # Ensure same shape
        if predictions.shape != self.mask.shape:
            # Resize predictions to match mask
            from skimage.transform import resize
            predictions = resize(predictions, self.mask.shape, order=0)

        # Binarize
        pred_bin = (predictions >= threshold).astype(np.uint8)

        # Calculate metrics
        tp = np.sum((pred_bin == 1) & (self.mask == 1))
        fp = np.sum((pred_bin == 1) & (self.mask == 0))
        fn = np.sum((pred_bin == 0) & (self.mask == 1))

        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }


def validate_with_ana(
    ana_path: Path,
    landsat_tile: Path,
    municipality: str = "Salto do Jacuí",
    state: str = "RS",
    year: int = 2023
) -> dict:
    """
    Complete validation pipeline: extract ANA mask, align with Landsat tile,
    and prepare for model validation.

    Args:
        ana_path: Path to ANA shapefiles
        landsat_tile: Path to Landsat tile (for reference)
        municipality: Target municipality
        state: Target state
        year: Target year

    Returns:
        Dictionary with validation data
    """
    logger.info("=" * 60)
    logger.info(f"ANA Validation for {municipality}/{state} ({year})")
    logger.info("=" * 60)

    # Initialize extractor
    extractor = ANAMaskExtractor(
        ana_path=ana_path,
        target_municipality=municipality,
        target_state=state,
        target_year=year
    )

    # Load and filter data
    extractor.load_ana_data()
    extractor.filter_by_municipality()

    # Create mask aligned with Landsat tile
    mask = extractor.create_pivot_mask(reference_raster=landsat_tile)

    # Save mask for later use
    output_dir = Path("data/validation/ana_masks")
    output_path = output_dir / \
        f"ana_mask_{municipality.replace(' ', '_')}_{year}.tif"
    extractor.save_mask(output_path)

    # Get pivot statistics
    pivot_info = []
    for idx, row in extractor.filtered_data.iterrows():
        info = {
            'id': row.get(extractor.ANA_COLUMNS['id'], idx),
            'area_ha': row.get(extractor.ANA_COLUMNS['area'], 0),
            'geometry_type': row.geometry.geom_type
        }
        if extractor.ANA_COLUMNS['radius'] in row:
            info['radius_m'] = row[extractor.ANA_COLUMNS['radius']]
        pivot_info.append(info)

    result = {
        'n_pivots': len(extractor.filtered_data),
        'mask_path': str(output_path),
        'mask_shape': mask.shape,
        'pivot_pixels': int(mask.sum()),
        'pivot_info': pivot_info,
        'bounds': {
            'minx': extractor.filtered_data.total_bounds[0],
            'miny': extractor.filtered_data.total_bounds[1],
            'maxx': extractor.filtered_data.total_bounds[2],
            'maxy': extractor.filtered_data.total_bounds[3]
        }
    }

    logger.info(
        f"Found {result['n_pivots']} pivots covering {result['pivot_pixels']} pixels")

    return result


# -----------------------------------------------------------------------------
# CLI INTERFACE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ANA masks for validation")
    parser.add_argument('--ana_path', type=str, default="data/raw/ana",
                        help="Path to ANA data directory")
    parser.add_argument('--landsat_tile', type=str, default="ibge/landsat_cpic_2023.tif",
                        help="Path to Landsat tile (for reference)")
    parser.add_argument('--municipality', type=str, default="Salto do Jacuí",
                        help="Target municipality")
    parser.add_argument('--state', type=str, default="RS",
                        help="Target state")
    parser.add_argument('--year', type=int, default=2023,
                        help="Target year")
    parser.add_argument('--output', type=str, default="data/validation",
                        help="Output directory")

    args = parser.parse_args()

    # Run validation
    result = validate_with_ana(
        ana_path=Path(args.ana_path),
        landsat_tile=Path(args.landsat_tile),
        municipality=args.municipality,
        state=args.state,
        year=args.year
    )

    print("\n" + "=" * 60)
    print("VALIDATION DATA READY")
    print("=" * 60)
    print(f"Pivots found: {result['n_pivots']}")
    print(f"Mask saved: {result['mask_path']}")
    print(f"Mask shape: {result['mask_shape']}")
    print(f"Pivot pixels: {result['pivot_pixels']}")
    print("\nTo use for validation:")
    print(f"  python main.py --mode validate --mask {result['mask_path']}")

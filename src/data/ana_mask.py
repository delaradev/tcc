import geopandas as gpd
import rasterio
import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from shapely.ops import unary_union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ANAMaskExtractor:
    ANA_COLUMNS = {
        'id': 'ID_PIVO', 'area': 'AREA_HA', 'state': 'UF',
        'municipality': 'MUNICIPIO', 'lat': 'LATITUDE', 'lon': 'LONGITUDE', 'radius': 'RAIO_M',
    }

    def __init__(self, ana_path: Path, target_municipality: str = "Salto do Jacuí",
                 target_state: str = "RS", target_year: int = 2023, buffer_meters: int = 100):
        self.ana_path = Path(ana_path)
        self.target_municipality = target_municipality
        self.target_state = target_state
        self.target_year = target_year
        self.buffer_meters = buffer_meters
        self.ana_data: Optional[gpd.GeoDataFrame] = None
        self.filtered_data: Optional[gpd.GeoDataFrame] = None
        self.mask: Optional[np.ndarray] = None

    def load_ana_data(self, shapefile: Optional[Path] = None) -> gpd.GeoDataFrame:
        if shapefile is None:
            shapefile_pattern = f"*{self.target_state}*{self.target_year}*.shp"
            shapefiles = list(self.ana_path.glob(shapefile_pattern))
            if not shapefiles:
                shapefiles = list(self.ana_path.glob(
                    f"*{self.target_state}*.shp"))
            if not shapefiles:
                raise FileNotFoundError(
                    f"No shapefile found for {self.target_state} in {self.ana_path}")
            shapefile = shapefiles[0]
            logger.info(f"Found shapefile: {shapefile.name}")

        logger.info(f"Loading ANA data from: {shapefile}")
        self.ana_data = gpd.read_file(shapefile)
        if self.ANA_COLUMNS['state'] in self.ana_data.columns:
            self.ana_data = self.ana_data[self.ana_data[self.ANA_COLUMNS['state']]
                                          == self.target_state]
            logger.info(
                f"Filtered by state {self.target_state}: {len(self.ana_data)} pivots")
        logger.info(f"Loaded {len(self.ana_data)} pivots")
        return self.ana_data

    def filter_by_municipality(self) -> gpd.GeoDataFrame:
        if self.ana_data is None:
            raise ValueError("Load ANA data first with load_ana_data()")
        possible_columns = ['MUNICIPIO', 'NM_MUN', 'MUN', 'MUNICIPIO_NM']
        mun_col = next(
            (col for col in possible_columns if col in self.ana_data.columns), None)
        if mun_col is None:
            logger.warning(
                f"Municipality column not found. Available: {self.ana_data.columns.tolist()}")
            return self.ana_data
        self.filtered_data = self.ana_data[self.ana_data[mun_col].str.contains(
            self.target_municipality, case=False, na=False)]
        logger.info(
            f"Found {len(self.filtered_data)} pivots in {self.target_municipality}")
        if len(self.filtered_data) == 0:
            logger.warning(f"No pivots found in {self.target_municipality}")
            municipalities = self.ana_data[mun_col].unique()[:20]
            logger.info(
                f"First 20 municipalities in {self.target_state}: {municipalities}")
        return self.filtered_data

    def create_pivot_mask(self, reference_raster: Optional[Path] = None,
                          output_resolution: float = 30.0, output_crs: str = 'EPSG:31982',
                          output_bounds: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        if self.filtered_data is None or len(self.filtered_data) == 0:
            raise ValueError(
                "No filtered data. Run filter_by_municipality() first.")
        geometries = []
        for _, row in self.filtered_data.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                radius = row.get(self.ANA_COLUMNS['radius'], 100)
                geometries.append(geom.buffer(radius))
            else:
                geometries.append(geom)
        union_geom = unary_union(geometries)

        if reference_raster and reference_raster.exists():
            with rasterio.open(reference_raster) as src:
                bounds = src.bounds
                transform = src.transform
                output_crs = src.crs
                width, height = src.width, src.height
            mask = features.rasterize([(union_geom, 1)], out_shape=(
                height, width), transform=transform, fill=0, dtype=np.uint8)
        elif output_bounds:
            minx, miny, maxx, maxy = output_bounds
            width = int((maxx - minx) / output_resolution)
            height = int((maxy - miny) / output_resolution)
            transform = from_origin(
                minx, maxy, output_resolution, output_resolution)
            mask = features.rasterize([(union_geom, 1)], out_shape=(
                height, width), transform=transform, fill=0, dtype=np.uint8)
        else:
            bounds = union_geom.bounds
            width = int((bounds[2] - bounds[0]) / output_resolution)
            height = int((bounds[3] - bounds[1]) / output_resolution)
            transform = from_origin(
                bounds[0], bounds[3], output_resolution, output_resolution)
            mask = features.rasterize([(union_geom, 1)], out_shape=(
                height, width), transform=transform, fill=0, dtype=np.uint8)

        self.mask = mask
        self.mask_transform = transform
        self.mask_crs = output_crs
        logger.info(
            f"Created mask: {mask.shape} pixels, {mask.sum()} pivot pixels")
        return mask

    def save_mask(self, output_path: Path, transform: Optional[rasterio.Affine] = None,
                  crs: Optional[str] = None) -> Path:
        if self.mask is None:
            raise ValueError("No mask created. Run create_pivot_mask() first.")
        transform = transform or self.mask_transform
        crs = crs or self.mask_crs
        if transform is None or crs is None:
            raise ValueError("Missing transform or CRS. Provide explicitly.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', driver='GTiff', height=self.mask.shape[0], width=self.mask.shape[1],
                           count=1, dtype=rasterio.uint8, crs=crs, transform=transform) as dst:
            dst.write(self.mask, 1)
        logger.info(f"Saved mask to: {output_path}")
        return output_path

    def compare_with_predictions(self, predictions: np.ndarray, threshold: float = 0.5) -> dict:
        if self.mask is None:
            raise ValueError("No mask created. Run create_pivot_mask() first.")
        if predictions.shape != self.mask.shape:
            from skimage.transform import resize
            predictions = resize(predictions, self.mask.shape, order=0)
        pred_bin = (predictions >= threshold).astype(np.uint8)
        tp = np.sum((pred_bin == 1) & (self.mask == 1))
        fp = np.sum((pred_bin == 1) & (self.mask == 0))
        fn = np.sum((pred_bin == 0) & (self.mask == 1))
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def validate_with_ana(ana_path: Path, landsat_tile: Path, municipality: str = "Salto do Jacuí",
                      state: str = "RS", year: int = 2023) -> dict:
    logger.info(f"ANA validation for {municipality}/{state} ({year})")
    extractor = ANAMaskExtractor(
        ana_path=ana_path, target_municipality=municipality, target_state=state, target_year=year)
    extractor.load_ana_data()
    extractor.filter_by_municipality()
    mask = extractor.create_pivot_mask(reference_raster=landsat_tile)
    output_dir = Path("data/validation/ana_masks")
    output_path = output_dir / \
        f"ana_mask_{municipality.replace(' ', '_')}_{year}.tif"
    extractor.save_mask(output_path)

    pivot_info = []
    for idx, row in extractor.filtered_data.iterrows():
        info = {'id': row.get(extractor.ANA_COLUMNS['id'], idx), 'area_ha': row.get(extractor.ANA_COLUMNS['area'], 0),
                'geometry_type': row.geometry.geom_type}
        if extractor.ANA_COLUMNS['radius'] in row:
            info['radius_m'] = row[extractor.ANA_COLUMNS['radius']]
        pivot_info.append(info)

    result = {
        'n_pivots': len(extractor.filtered_data), 'mask_path': str(output_path), 'mask_shape': mask.shape,
        'pivot_pixels': int(mask.sum()), 'pivot_info': pivot_info,
        'bounds': {
            'minx': extractor.filtered_data.total_bounds[0], 'miny': extractor.filtered_data.total_bounds[1],
            'maxx': extractor.filtered_data.total_bounds[2], 'maxy': extractor.filtered_data.total_bounds[3]
        }
    }
    logger.info(
        f"Found {result['n_pivots']} pivots covering {result['pivot_pixels']} pixels")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract ANA masks for validation")
    parser.add_argument('--ana_path', type=str,
                        default="data/raw/ana", help="Path to ANA data directory")
    parser.add_argument('--landsat_tile', type=str,
                        default="ibge/landsat_cpic_2023.tif", help="Path to Landsat tile")
    parser.add_argument('--municipality', type=str,
                        default="Salto do Jacuí", help="Target municipality")
    parser.add_argument('--state', type=str, default="RS", help="Target state")
    parser.add_argument('--year', type=int, default=2023, help="Target year")
    parser.add_argument('--output', type=str,
                        default="data/validation", help="Output directory")
    args = parser.parse_args()

    result = validate_with_ana(ana_path=Path(args.ana_path), landsat_tile=Path(args.landsat_tile),
                               municipality=args.municipality, state=args.state, year=args.year)
    print(
        f"Validation data ready. Pivots: {result['n_pivots']}, mask: {result['mask_path']}, shape: {result['mask_shape']}")

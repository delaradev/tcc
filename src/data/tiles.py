"""
================================================================================
TILE GENERATOR FOR LANDSAT CPIC COMPOSITES
================================================================================

Generates standardized image tiles from Landsat-derived CPIC composites
for subsequent deep learning model training.

Input: 3-band GeoTIFF [EVI_max, BSI_max, GREEN_median] at 30m resolution
Output: 512x512 PNG tiles with configurable overlap and normalization

Based on methodology from:
Liu et al. (2023) - ISPRS Journal of Photogrammetry and Remote Sensing

Author: Emanuel de Lara Ruas
Version: 1.0.0
================================================================================
"""

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

# -----------------------------------------------------------------------------
# MODULE CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NORMALIZATION STRATEGIES
# -----------------------------------------------------------------------------

class NormalizationStrategy:
    """
    Implements spectral band normalization strategies described in Liu et al. (2023)

    Two approaches:
        - 'global': Fixed percentile ranges derived from literature
        - 'adaptive': Dataset-specific percentile-based normalization
    """

    # Band-specific normalization bounds from Liu et al. (2023), Section 3.1
    BAND_BOUNDS = {
        0: {'name': 'EVI_max', 'low': 0.0, 'high': 0.7},
        1: {'name': 'BSI_max', 'low': -0.1, 'high': 0.3},
        2: {'name': 'GREEN_median', 'low': 0.0, 'high': 0.15}
    }

    @staticmethod
    def _percentile_normalize(
        band: np.ndarray,
        p_low: float = 2,
        p_high: float = 98
    ) -> np.ndarray:
        """
        Normalize band using percentile clipping to remove outliers

        Args:
            band: Input band array (float32)
            p_low: Lower percentile for clipping
            p_high: Upper percentile for clipping

        Returns:
            Normalized band in [0, 1] range
        """
        valid_pixels = band[~np.isnan(band)]

        if len(valid_pixels) == 0:
            return np.zeros_like(band)

        low_val = np.percentile(valid_pixels, p_low)
        high_val = np.percentile(valid_pixels, p_high)

        if high_val <= low_val:
            return np.zeros_like(band)

        normalized = (band - low_val) / (high_val - low_val)
        return np.clip(normalized, 0.0, 1.0)

    @classmethod
    def apply(
        cls,
        stack: np.ndarray,
        method: str = 'global'
    ) -> np.ndarray:
        """
        Apply normalization to multi-band stack

        Args:
            stack: Array of shape (height, width, bands)
            method: Normalization method ('global' or 'percentile')

        Returns:
            Normalized uint8 array ready for PNG export
        """
        normalized = np.zeros_like(stack, dtype=np.float32)
        n_bands = stack.shape[2]

        for band_idx in range(n_bands):
            band = stack[:, :, band_idx]

            if method == 'global':
                bounds = cls.BAND_BOUNDS.get(band_idx)
                if bounds is None:
                    logger.warning(f"No bounds defined for band {band_idx}")
                    band_norm = cls._percentile_normalize(band)
                else:
                    band_norm = (band - bounds['low']) / \
                        (bounds['high'] - bounds['low'])
                    band_norm = np.clip(band_norm, 0.0, 1.0)
            else:
                band_norm = cls._percentile_normalize(band)

            normalized[:, :, band_idx] = band_norm

        return (normalized * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# TILE GENERATION
# -----------------------------------------------------------------------------

class TileGenerator:
    """
    Generates overlapping tiles from GeoTIFF for model training

    Implements sliding window with configurable overlap to avoid edge artifacts
    during inference, as recommended in Liu et al. (2023), Section 3.4
    """

    def __init__(
        self,
        source_path: Path,
        output_dir: Path,
        tile_size: int = 512,
        overlap: int = 32,
        normalize_method: str = 'global'
    ):
        """
        Initialize tile generator

        Args:
            source_path: Path to input GeoTIFF
            output_dir: Directory for output PNG tiles
            tile_size: Square tile dimension in pixels
            overlap: Overlap between adjacent tiles
            normalize_method: Normalization strategy ('global' or 'percentile')
        """
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.normalize_method = normalize_method

        # Validate inputs
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {self.source_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata to be populated during processing
        self.metadata: Dict = {}

    def _compute_tile_grid(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Compute tile grid dimensions with overlap

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Tuple of (n_tiles_x, n_tiles_y, step_x, step_y)
        """
        step = self.tile_size - self.overlap
        n_tiles_x = (width - self.tile_size + step - 1) // step + 1
        n_tiles_y = (height - self.tile_size + step - 1) // step + 1

        return n_tiles_x, n_tiles_y, step, step

    def _extract_tile(
        self,
        src: rasterio.DatasetReader,
        x: int,
        y: int
    ) -> np.ndarray:
        """
        Extract a single tile from the source image

        Args:
            src: Open rasterio dataset
            x: X coordinate (column) of tile origin
            y: Y coordinate (row) of tile origin

        Returns:
            Tile array of shape (tile_size, tile_size, bands)
        """
        window = Window(x, y, self.tile_size, self.tile_size)
        tile_data = src.read(window=window)  # Shape: (bands, H, W)
        return np.transpose(tile_data, (1, 2, 0))  # Shape: (H, W, bands)

    def generate(self) -> Tuple[int, Dict]:
        """
        Generate all tiles from source image

        Returns:
            Tuple of (total_tiles_count, metadata)
        """
        logger.info(f"Processing source: {self.source_path}")

        with rasterio.open(self.source_path) as src:
            width = src.width
            height = src.height
            bands = src.count

            logger.info(f"Image dimensions: {width} x {height} x {bands}")
            logger.info(f"Resolution: {src.res} meters/pixel")
            logger.info(f"CRS: {src.crs}")

            if bands != 3:
                logger.warning(f"Expected 3 bands, found {bands}")

            n_tiles_x, n_tiles_y, step_x, step_y = self._compute_tile_grid(
                width, height)
            total_tiles = n_tiles_x * n_tiles_y

            logger.info(
                f"Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles"
            )
            logger.info(f"Step: {step_x}px, Overlap: {self.overlap}px")

            tile_counter = 0

            for y_idx in range(n_tiles_y):
                y = min(y_idx * step_y, height - self.tile_size)

                for x_idx in range(n_tiles_x):
                    x = min(x_idx * step_x, width - self.tile_size)

                    tile_data = self._extract_tile(src, x, y)

                    # Apply normalization
                    normalized = NormalizationStrategy.apply(
                        tile_data,
                        method=self.normalize_method
                    )

                    # Save as PNG
                    output_path = self.output_dir / \
                        f"tile_{tile_counter:05d}.png"
                    Image.fromarray(normalized).save(output_path)

                    tile_counter += 1

                    if tile_counter % 100 == 0:
                        logger.info(
                            f"Generated {tile_counter}/{total_tiles} tiles")

            # Store metadata
            self.metadata = {
                'source': str(self.source_path),
                'width': width,
                'height': height,
                'bands': bands,
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'total_tiles': total_tiles,
                'normalize_method': self.normalize_method,
                'resolution_meters': src.res[0],
                'crs': str(src.crs)
            }

            logger.info(f"Generation complete: {total_tiles} tiles created")

            return total_tiles, self.metadata

    def validate(self) -> Dict:
        """
        Validate tile generation completeness and quality

        Returns:
            Validation statistics dictionary
        """
        tiles = sorted(self.output_dir.glob("tile_*.png"))
        actual_count = len(tiles)
        expected_count = self.metadata.get('total_tiles', 0)

        validation = {
            'expected_tiles': expected_count,
            'actual_tiles': actual_count,
            'complete': actual_count == expected_count,
            'difference': expected_count - actual_count
        }

        logger.info("Validation results:")
        logger.info(f"  Expected tiles: {expected_count}")
        logger.info(f"  Actual tiles: {actual_count}")

        if validation['complete']:
            logger.info("  All tiles successfully generated")
        else:
            logger.warning(
                f"  Missing {validation['difference']} tiles"
            )

        return validation

    def compute_statistics(self, n_samples: int = 100) -> Dict:
        """
        Compute statistical summary of generated tiles

        Args:
            n_samples: Number of tiles to sample

        Returns:
            Dictionary with band-wise statistics
        """
        tiles = sorted(self.output_dir.glob("tile_*.png"))[:n_samples]

        if not tiles:
            logger.warning("No tiles found for statistics computation")
            return {}

        band_means = []
        band_stds = []

        for tile_path in tiles:
            img = Image.open(tile_path)
            arr = np.array(img) / 255.0
            band_means.append(arr.mean(axis=(0, 1)))
            band_stds.append(arr.std(axis=(0, 1)))

        band_means = np.array(band_means)
        band_stds = np.array(band_stds)

        stats = {
            'n_samples': len(tiles),
            'band_means': band_means.mean(axis=0).tolist(),
            'band_stds': band_stds.mean(axis=0).tolist(),
            'band_means_range': [
                (band_means[:, i].min(), band_means[:, i].max())
                for i in range(3)
            ]
        }

        logger.info(f"Statistics from {len(tiles)} sampled tiles:")
        for b, name in enumerate(['EVI_max', 'BSI_max', 'GREEN_median']):
            logger.info(
                f"  {name}: mean={stats['band_means'][b]:.3f}, "
                f"std={stats['band_stds'][b]:.3f}"
            )

        return stats


# -----------------------------------------------------------------------------
# CLI INTERFACE
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate normalized tiles from Landsat CPIC composites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tif_path data/landsat_cpic_2023.tif --out_dir tiles/images
  %(prog)s --tif_path data/landsat_cpic_2023.tif --tile_size 256 --overlap 16
  %(prog)s --tif_path data/landsat_cpic_2023.tif --normalize percentile --validate
        """
    )

    parser.add_argument(
        '--tif_path',
        type=str,
        default='ibge/landsat_cpic_2023.tif',
        help='Path to input GeoTIFF file'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='tiles/images',
        help='Output directory for PNG tiles'
    )

    parser.add_argument(
        '--tile_size',
        type=int,
        default=512,
        help='Tile size in pixels (default: 512)'
    )

    parser.add_argument(
        '--overlap',
        type=int,
        default=32,
        help='Overlap between adjacent tiles in pixels (default: 32)'
    )

    parser.add_argument(
        '--normalize',
        type=str,
        default='global',
        choices=['global', 'percentile'],
        help='Normalization method (default: global)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate tile generation after completion'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Compute statistics from generated tiles'
    )

    parser.add_argument(
        '--sample_size',
        type=int,
        default=100,
        help='Number of tiles for statistics sampling (default: 100)'
    )

    return parser.parse_args()


def main() -> None:
    """Main execution entry point"""
    args = parse_arguments()

    try:
        generator = TileGenerator(
            source_path=Path(args.tif_path),
            output_dir=Path(args.out_dir),
            tile_size=args.tile_size,
            overlap=args.overlap,
            normalize_method=args.normalize
        )

        total_tiles, metadata = generator.generate()

        if args.validate:
            generator.validate()

        if args.stats:
            generator.compute_statistics(n_samples=args.sample_size)

        logger.info(f"\nTile generation completed successfully")
        logger.info(f"   Output directory: {args.out_dir}")
        logger.info(f"   Total tiles: {total_tiles}")

    except Exception as e:
        logger.error(f"Tile generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

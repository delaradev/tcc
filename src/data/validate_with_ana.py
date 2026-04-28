#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VALIDATE MODEL WITH ANA MASKS
================================================================================

Compare model predictions with official ANA masks for the same region.

Author: Emanuel de Lara Ruas
Version: 1.0.0
================================================================================
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ANAValidator:
    """Validate model predictions against ANA ground truth masks"""

    def __init__(self, ana_mask_path: Path, landsat_tile_path: Path):
        """
        Initialize validator

        Args:
            ana_mask_path: Path to ANA mask (created by extractor)
            landsat_tile_path: Path to Landsat tile used for prediction
        """
        self.ana_mask_path = Path(ana_mask_path)
        self.landsat_tile_path = Path(landsat_tile_path)

        self.ana_mask: Optional[np.ndarray] = None
        self.landsat_data: Optional[np.ndarray] = None

    def load_masks(self) -> None:
        """Load ANA mask and Landsat tile"""
        # Load ANA mask
        with rasterio.open(self.ana_mask_path) as src:
            self.ana_mask = src.read(1)
            self.ana_transform = src.transform
            self.ana_crs = src.crs
            self.ana_bounds = src.bounds
            logger.info(f"ANA mask: {self.ana_mask.shape} pixels, "
                        f"{self.ana_mask.sum():.0f} pivot pixels")

        # Load Landsat tile
        with rasterio.open(self.landsat_tile_path) as src:
            self.landsat_data = src.read()
            self.landsat_transform = src.transform
            logger.info(f"Landsat tile: {self.landsat_data.shape}")

    def load_predictions(self, predictions_path: Path) -> np.ndarray:
        """Load model predictions"""
        with rasterio.open(predictions_path) as src:
            pred = src.read(1)
            logger.info(f"Predictions shape: {pred.shape}")
            return pred

    def compute_metrics(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute validation metrics

        Args:
            predictions: Model predictions (0-1 probability)
            threshold: Classification threshold

        Returns:
            Dictionary with metrics
        """
        if self.ana_mask is None:
            self.load_masks()

        # Ensure same shape
        if predictions.shape != self.ana_mask.shape:
            from skimage.transform import resize
            logger.warning(
                f"Resizing predictions from {predictions.shape} to {self.ana_mask.shape}")
            predictions = resize(predictions, self.ana_mask.shape, order=0)

        # Binarize predictions
        pred_bin = (predictions >= threshold).astype(np.uint8)

        # Calculate metrics
        tp = np.sum((pred_bin == 1) & (self.ana_mask == 1))
        fp = np.sum((pred_bin == 1) & (self.ana_mask == 0))
        fn = np.sum((pred_bin == 0) & (self.ana_mask == 1))
        tn = np.sum((pred_bin == 0) & (self.ana_mask == 0))

        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_pixels': self.ana_mask.size
        }

    def visualize_comparison(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[Path] = None
    ) -> None:
        """Visualize comparison between predictions and ANA mask"""
        if self.ana_mask is None:
            self.load_masks()

        # Ensure same shape
        if predictions.shape != self.ana_mask.shape:
            from skimage.transform import resize
            predictions = resize(predictions, self.ana_mask.shape, order=0)

        pred_bin = (predictions >= threshold).astype(np.uint8)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # ANA ground truth
        ax = axes[0, 0]
        im = ax.imshow(self.ana_mask, cmap='Greens', interpolation='nearest')
        ax.set_title(
            f'ANA Ground Truth ({self.ana_mask.sum():.0f} pivot pixels)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # Model predictions
        ax = axes[0, 1]
        im = ax.imshow(predictions, cmap='hot', vmin=0,
                       vmax=1, interpolation='nearest')
        ax.set_title(f'Model Predictions (probability)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # Binarized predictions
        ax = axes[1, 0]
        im = ax.imshow(pred_bin, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Binarized Predictions (threshold={threshold})')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # Overlay comparison
        ax = axes[1, 1]

        # Create RGB overlay
        overlay = np.zeros((*self.ana_mask.shape, 3), dtype=np.uint8)
        # Green: ANA mask
        overlay[:, :, 1] = (self.ana_mask * 255).astype(np.uint8)
        # Red: Model predictions
        overlay[:, :, 0] = (pred_bin * 255).astype(np.uint8)
        # Yellow: Overlap (both)

        ax.imshow(overlay)
        ax.set_title('Overlay (Green=ANA, Red=Prediction, Yellow=Both)')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to: {save_path}")

        plt.show()

    def run_validation(
        self,
        predictions_path: Path,
        threshold: float = 0.5,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """Run complete validation"""
        logger.info("=" * 60)
        logger.info("Validating Model with ANA Masks")
        logger.info("=" * 60)

        # Load predictions
        predictions = self.load_predictions(predictions_path)

        # Compute metrics
        metrics = self.compute_metrics(predictions, threshold)

        # Print results
        logger.info("\n📊 Validation Results:")
        logger.info(f"  IoU:        {metrics['iou']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:   {metrics['f1']:.4f}")
        logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"\n  True Positives:  {metrics['tp']}")
        logger.info(f"  False Positives: {metrics['fp']}")
        logger.info(f"  False Negatives: {metrics['fn']}")
        logger.info(f"  Total Pixels:    {metrics['total_pixels']}")

        # Visualize
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / 'validation_comparison.png'
            self.visualize_comparison(predictions, threshold, save_path)
        else:
            self.visualize_comparison(predictions, threshold)

        return metrics


# -----------------------------------------------------------------------------
# CLI INTERFACE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate model with ANA masks")
    parser.add_argument('--ana_mask', type=str, required=True,
                        help="Path to ANA mask (created by extractor)")
    parser.add_argument('--landsat_tile', type=str, default="ibge/landsat_cpic_2023.tif",
                        help="Path to Landsat tile")
    parser.add_argument('--predictions', type=str, required=True,
                        help="Path to model predictions (GeoTIFF)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument('--output_dir', type=str, default="data/validation",
                        help="Output directory for visualizations")

    args = parser.parse_args()

    validator = ANAValidator(
        ana_mask_path=Path(args.ana_mask),
        landsat_tile_path=Path(args.landsat_tile)
    )

    metrics = validator.run_validation(
        predictions_path=Path(args.predictions),
        threshold=args.threshold,
        output_dir=Path(args.output_dir)
    )

    print("\n✅ Validation completed!")

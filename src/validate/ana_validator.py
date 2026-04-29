import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ANAValidator:
    def __init__(self, ana_mask_path: Path, landsat_tile_path: Path):
        self.ana_mask_path = Path(ana_mask_path)
        self.landsat_tile_path = Path(landsat_tile_path)
        self.ana_mask: Optional[np.ndarray] = None
        self.landsat_data: Optional[np.ndarray] = None

    def load_masks(self) -> None:
        with rasterio.open(self.ana_mask_path) as src:
            self.ana_mask = src.read(1)
            self.ana_transform = src.transform
            self.ana_crs = src.crs
            self.ana_bounds = src.bounds
            logger.info(
                f"ANA mask: {self.ana_mask.shape} pixels, {self.ana_mask.sum():.0f} pivot pixels")
        with rasterio.open(self.landsat_tile_path) as src:
            self.landsat_data = src.read()
            self.landsat_transform = src.transform
            logger.info(f"Landsat tile: {self.landsat_data.shape}")

    def load_predictions(self, predictions_path: Path) -> np.ndarray:
        with rasterio.open(predictions_path) as src:
            pred = src.read(1)
            logger.info(f"Predictions shape: {pred.shape}")
            return pred

    def compute_metrics(self, predictions: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        if self.ana_mask is None:
            self.load_masks()
        if predictions.shape != self.ana_mask.shape:
            from skimage.transform import resize
            logger.warning(
                f"Resizing predictions from {predictions.shape} to {self.ana_mask.shape}")
            predictions = resize(predictions, self.ana_mask.shape, order=0)
        pred_bin = (predictions >= threshold).astype(np.uint8)
        tp = np.sum((pred_bin == 1) & (self.ana_mask == 1))
        fp = np.sum((pred_bin == 1) & (self.ana_mask == 0))
        fn = np.sum((pred_bin == 0) & (self.ana_mask == 1))
        tn = np.sum((pred_bin == 0) & (self.ana_mask == 0))
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn), 'total_pixels': self.ana_mask.size}

    def visualize_comparison(self, predictions: np.ndarray, threshold: float = 0.5, save_path: Optional[Path] = None) -> None:
        if self.ana_mask is None:
            self.load_masks()
        if predictions.shape != self.ana_mask.shape:
            from skimage.transform import resize
            predictions = resize(predictions, self.ana_mask.shape, order=0)
        pred_bin = (predictions >= threshold).astype(np.uint8)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(self.ana_mask, cmap='Greens',
                          interpolation='nearest')
        axes[0, 0].set_title(
            f'ANA Ground Truth ({self.ana_mask.sum():.0f} pivot pixels)')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(predictions, cmap='hot', vmin=0,
                          vmax=1, interpolation='nearest')
        axes[0, 1].set_title('Model Predictions (probability)')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(pred_bin, cmap='Blues', interpolation='nearest')
        axes[1, 0].set_title(f'Binarized Predictions (threshold={threshold})')
        axes[1, 0].axis('off')
        overlay = np.zeros((*self.ana_mask.shape, 3), dtype=np.uint8)
        overlay[:, :, 1] = (self.ana_mask * 255).astype(np.uint8)
        overlay[:, :, 0] = (pred_bin * 255).astype(np.uint8)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(
            'Overlay (Green=ANA, Red=Prediction, Yellow=Both)')
        axes[1, 1].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to: {save_path}")
        plt.show()

    def run_validation(self, predictions_path: Path, threshold: float = 0.5, output_dir: Optional[Path] = None) -> Dict:
        logger.info("Validating model with ANA masks")
        predictions = self.load_predictions(predictions_path)
        metrics = self.compute_metrics(predictions, threshold)
        logger.info("Validation Results:")
        for key in ['iou', 'precision', 'recall', 'f1', 'accuracy']:
            logger.info(f"  {key}: {metrics[key]:.4f}")
        logger.info(
            f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, Total pixels: {metrics['total_pixels']}")
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.visualize_comparison(
                predictions, threshold, output_dir / 'validation_comparison.png')
        else:
            self.visualize_comparison(predictions, threshold)
        return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate model with ANA masks")
    parser.add_argument('--ana_mask', type=str,
                        required=True, help="Path to ANA mask")
    parser.add_argument('--landsat_tile', type=str,
                        default="ibge/landsat_cpic_2023.tif", help="Path to Landsat tile")
    parser.add_argument('--predictions', type=str, required=True,
                        help="Path to model predictions (GeoTIFF)")
    parser.add_argument('--threshold', type=float,
                        default=0.5, help="Classification threshold")
    parser.add_argument('--output_dir', type=str,
                        default="data/validation", help="Output directory")
    args = parser.parse_args()

    validator = ANAValidator(ana_mask_path=Path(
        args.ana_mask), landsat_tile_path=Path(args.landsat_tile))
    metrics = validator.run_validation(predictions_path=Path(
        args.predictions), threshold=args.threshold, output_dir=Path(args.output_dir))
    print("Validation completed.")

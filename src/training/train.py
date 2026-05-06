import json
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from src.models.unet import build_unet
from src.models.losses import tversky_loss
from src.training.metrics import iou_score, dice_score, precision_score, recall_score
from src.training.callbacks import PredictionSaver, EpochVisualizationCallback
from src.data.dataset_balancer import CPICDatasetBuilder
from src.utils.gpu_utils import configure_gpu, get_gpu_info
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.run_name = f"{self.config['project']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_output = os.environ.get('CPIC_OUTPUT_DIR', 'runs')
        self.output_dir = Path(base_output) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gpu_config = self.config['gpu']
        configure_gpu(
            memory_growth=gpu_config['memory_growth'],
            memory_limit_mb=gpu_config.get('memory_limit_mb'),
            mixed_precision=gpu_config['mixed_precision']
        )

        logger.info(f"Run: {self.run_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"GPU: {get_gpu_info()}")
        self.resume_from: Optional[str] = None

    def prepare_data(self):
        data_config = self.config['data']
        model_config = self.config['model']

        max_samples = data_config.get('max_train_samples', None)

        self.dataset_builder = CPICDatasetBuilder(
            dataset_path=data_config['balanced_path'],
            image_size=model_config['img_size'],
            seed=self.config['project']['seed'],
            max_train_samples=max_samples
        )

        train_pairs = self.dataset_builder.load_pairs('train')
        val_pairs = self.dataset_builder.load_pairs('valid')
        logger.info(
            f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

        self.train_ds = self.dataset_builder.build_dataset(
            split='train',
            batch_size=self.config['training']['batch_size'],
            training=True
        )
        self.val_ds = self.dataset_builder.build_dataset(
            split='valid',
            batch_size=self.config['training']['batch_size'],
            training=False
        )
        self.sample_ds = self.val_ds.take(1)

    def build_model(self):
        model_config = self.config['model']
        self.model = build_unet(
            input_shape=(
                model_config['img_size'],
                model_config['img_size'],
                model_config['input_channels']
            ),
            base_filters=model_config['unet_base_filters']
        )
        logger.info(self.model.summary())

    def compile_model(self):
        train_config = self.config['training']
        loss_config = train_config['loss']

        loss_fn = tversky_loss(
            alpha=loss_config['alpha'], beta=loss_config['beta'])

        metrics = [
            iou_score(),
            dice_score(),
            precision_score(),
            recall_score()
        ]

        if train_config['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=train_config['learning_rate'])
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=train_config['learning_rate'])

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def setup_callbacks(self):
        train_config = self.config['training']

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model.keras'),
                monitor='val_iou_score',
                mode='max',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=train_config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=train_config['reduce_lr_patience'],
                min_lr=1e-7
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.output_dir / 'tensorboard'),
            PredictionSaver(self.sample_ds, str(
                self.output_dir / 'predictions'), max_samples=4),
            EpochVisualizationCallback(
                validation_ds=self.val_ds,
                output_dir=str(self.output_dir / 'epoch_vis'),
                num_samples=9
            )
        ]
        self.callbacks = callbacks

    def train(self):
        train_config = self.config['training']

        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        logger.info("Starting training")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=train_config['epochs'],
            callbacks=self.callbacks,
            verbose=1
        )

        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history.history, f, indent=2)

        self.post_training_analysis(history.history)

        logger.info(f"Training completed. Model saved in {self.output_dir}")

    # -------------------------------------------------------------------------
    # Análises pós-treinamento (solicitadas pelo professor)
    # -------------------------------------------------------------------------
    def post_training_analysis(self, history: dict):
        logger.info("Running post-training analysis...")
        analysis_dir = self.output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)

        self.plot_metrics_history(history, analysis_dir)

        best_model_path = self.output_dir / 'best_model.keras'
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            best_model = tf.keras.models.load_model(
                best_model_path,
                custom_objects={
                    'tversky_loss': tversky_loss(
                        alpha=self.config['training']['loss']['alpha'],
                        beta=self.config['training']['loss']['beta']
                    ),
                    'iou_score': iou_score(),
                    'dice_score': dice_score(),
                    'precision_score': precision_score(),
                    'recall_score': recall_score()
                }
            )
        else:
            logger.warning("Best model not found, using current model")
            best_model = self.model

        img_metrics = self.evaluate_per_image(best_model, analysis_dir)
        self.save_ranking_and_visualizations(img_metrics, analysis_dir)

    def plot_metrics_history(self, history: dict, output_dir: Path):
        metrics = ['loss', 'iou_score', 'dice_score',
                   'precision_score', 'recall_score']
        epochs = len(history.get('loss', []))
        if epochs == 0:
            return

        step = max(5, epochs // 20)
        indices = list(range(0, epochs, step))
        if indices[-1] != epochs - 1:
            indices.append(epochs - 1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
            ax = axes[idx]
            train_vals = [history[metric][i] for i in indices]
            ax.plot(indices, train_vals, 'b-o', label='Train', markersize=4)
            val_key = f'val_{metric}'
            if val_key in history:
                val_vals = [history[val_key][i] for i in indices]
                ax.plot(indices, val_vals, 'r-s',
                        label='Validation', markersize=4)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_evolution.png', dpi=150)
        plt.close()
        logger.info(
            f"Metrics plot saved to {output_dir / 'metrics_evolution.png'}")

    def evaluate_per_image(self, model: tf.keras.Model, output_dir: Path) -> List[dict]:
        val_pairs = self.dataset_builder.load_pairs('valid')
        if not val_pairs:
            logger.warning(
                "No validation pairs found for per-image evaluation")
            return []

        img_size = self.config['model']['img_size']
        results = []

        for img_path, mask_path in val_pairs:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, axis=0)

            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.resize(mask, [img_size, img_size])
            mask = tf.cast(mask, tf.float32) / 255.0
            mask_np = mask.numpy().squeeze()

            pred = model.predict(img, verbose=0)[0, :, :, 0]
            pred_bin = (pred >= 0.5).astype(np.float32)

            intersection = np.sum((pred_bin == 1) & (mask_np == 1))
            union = np.sum((pred_bin == 1) | (mask_np == 1))
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / \
                (np.sum(pred_bin) + np.sum(mask_np) + 1e-8)
            precision = intersection / (np.sum(pred_bin) + 1e-8)
            recall = intersection / (np.sum(mask_np) + 1e-8)

            results.append({
                'img_name': Path(img_path).stem,
                'img_path': str(img_path),
                'mask_path': str(mask_path),
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(precision),
                'recall': float(recall),
                'prediction': pred,
                'mask': mask_np,
                'image': img.numpy().squeeze()
            })
        return results

    def save_ranking_and_visualizations(self, img_metrics: List[dict], output_dir: Path):
        if not img_metrics:
            return

        sorted_by_iou = sorted(img_metrics, key=lambda x: x['iou'])
        worst_5 = sorted_by_iou[:5]
        best_5 = sorted_by_iou[-5:]

        ranking = {
            'best_5': [
                {k: v for k, v in m.items() if k not in [
                    'prediction', 'mask', 'image']}
                for m in best_5
            ],
            'worst_5': [
                {k: v for k, v in m.items() if k not in [
                    'prediction', 'mask', 'image']}
                for m in worst_5
            ]
        }
        with open(output_dir / 'ranking_top5_bottom5.json', 'w') as f:
            json.dump(ranking, f, indent=2)
        logger.info(
            f"Ranking saved to {output_dir / 'ranking_top5_bottom5.json'}")

        targets = [0.0, 0.25, 0.5, 0.75, 1.0]
        samples_for_vis = best_5 + worst_5
        for target in targets:
            closest = min(img_metrics, key=lambda x: abs(x['iou'] - target))
            if closest not in samples_for_vis:
                samples_for_vis.append(closest)

        seen = set()
        unique_samples = []
        for m in samples_for_vis:
            if m['img_name'] not in seen:
                seen.add(m['img_name'])
                unique_samples.append(m)

        vis_dir = output_dir / 'sample_visualizations'
        vis_dir.mkdir(exist_ok=True)
        for sample in unique_samples:
            self.visualize_overlay(sample, vis_dir)
        logger.info(f"Visualizations saved to {vis_dir}")

    def visualize_overlay(self, sample: dict, output_dir: Path):
        img = sample['image']
        mask = sample['mask']
        pred_bin = (sample['prediction'] >= 0.5).astype(np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title(f'Ground Truth (IoU: {sample["iou"]:.3f})')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(pred_bin, cmap='gray')
        axes[1, 0].set_title('Prediction (Binary)')
        axes[1, 0].axis('off')
        overlay = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        tp = (mask == 1) & (pred_bin == 1)
        fp = (mask == 0) & (pred_bin == 1)
        fn = (mask == 1) & (pred_bin == 0)
        overlay[tp, 1] = 1.0
        overlay[fp, 0] = 1.0
        overlay[fn, 2] = 1.0
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Green=TP, Red=FP, Blue=FN)')
        axes[1, 1].axis('off')
        plt.suptitle(
            f"{sample['img_name']} | IoU={sample['iou']:.3f} | Dice={sample['dice']:.3f}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{sample['img_name']}_iou_{sample['iou']:.3f}.png", dpi=100)
        plt.close()

    def run(self):
        self.prepare_data()
        self.build_model()
        self.compile_model()
        self.setup_callbacks()
        self.train()

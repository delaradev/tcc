import json
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import tensorflow as tf

from src.models.unet import build_unet
from src.models.losses import tversky_loss
from src.training.metrics import iou_score, dice_score, precision_score, recall_score
from src.training.callbacks import PredictionSaver
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

        logger.info(f"Training completed. Model saved in {self.output_dir}")

    def run(self):
        self.prepare_data()
        self.build_model()
        self.compile_model()
        self.setup_callbacks()
        self.train()

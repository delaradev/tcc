import tensorflow as tf
from pathlib import Path
import yaml
import json
from datetime import datetime

from src.models.unet import build_unet
from src.models.losses import tversky_loss
from src.training.metrics import iou_score, dice_score, precision_score, recall_score
from src.training.callbacks import PredictionSaver, GPUMemoryMonitor
from src.dataset_builder import CPICDataset
from src.utils.gpu_utils import configure_gpu, get_gpu_info


class Trainer:
    """Classe principal para treinamento"""

    def __init__(self, config_path: str):
        """Inicializa treinador com configurações"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.run_name = f"{self.config['project']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path('runs') / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configurar GPU
        gpu_config = self.config['gpu']
        configure_gpu(
            memory_growth=gpu_config['memory_growth'],
            memory_limit_mb=gpu_config.get('memory_limit_mb'),
            mixed_precision=gpu_config['mixed_precision']
        )

        print(f"Run: {self.run_name}")
        print(f"GPU Info: {get_gpu_info()}")

    def prepare_data(self):
        """Prepara datasets de treino e validação"""
        data_config = self.config['data']
        model_config = self.config['model']

        dataset = CPICDataset(
            data_config['dataset_path'], model_config['img_size'])

        # Carregar pares
        train_pairs = dataset.load_pairs('train')
        val_pairs = dataset.load_pairs('val')

        print(f"Dados: treino={len(train_pairs)}, val={len(val_pairs)}")

        # Criar datasets
        self.train_ds = dataset.create_dataset(
            train_pairs,
            batch_size=self.config['training']['batch_size'],
            training=True,
            randommix=self.config['training']['randommix'],
            randommix_prob=self.config['training']['randommix_prob']
        )

        self.val_ds = dataset.create_dataset(
            val_pairs,
            batch_size=self.config['training']['batch_size'],
            training=False,
            randommix=False
        )

        # Guardar uma amostra para visualização
        self.sample_ds = self.val_ds.take(1)

    def build_model(self):
        """Constrói o modelo U-Net"""
        model_config = self.config['model']

        self.model = build_unet(
            input_shape=(model_config['img_size'],
                         model_config['img_size'],
                         model_config['input_channels']),
            base_filters=model_config['unet_base_filters']
        )

        print(self.model.summary())

    def compile_model(self):
        """Compila o modelo com loss e métricas"""
        train_config = self.config['training']
        loss_config = train_config['loss']

        # Loss
        loss_fn = tversky_loss(
            alpha=loss_config['alpha'],
            beta=loss_config['beta']
        )

        # Métricas
        metrics = [
            iou_score(),
            dice_score(),
            precision_score(),
            recall_score()
        ]

        # Otimizador
        if train_config['optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=train_config['learning_rate'])
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=train_config['learning_rate'])

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

    def setup_callbacks(self):
        """Configura callbacks"""
        train_config = self.config['training']

        callbacks = []

        # Checkpoint
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(self.output_dir / 'best_model.keras'),
            monitor='val_iou_score',
            mode='max',
            save_best_only=True
        ))

        # Early stopping
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=train_config['early_stopping_patience'],
            restore_best_weights=True
        ))

        # Reduce LR on plateau
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=train_config['reduce_lr_patience'],
            min_lr=1e-7
        ))

        # TensorBoard
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=self.output_dir / 'tensorboard'
        ))

        # Predição samples
        callbacks.append(PredictionSaver(
            self.sample_ds,
            str(self.output_dir / 'predictions'),
            max_samples=4
        ))

        # Monitor GPU
        callbacks.append(GPUMemoryMonitor())

        self.callbacks = callbacks

    def train(self):
        """Executa treinamento"""
        train_config = self.config['training']

        # Salvar configuração
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        print("\nIniciando treinamento...")

        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=train_config['epochs'],
            callbacks=self.callbacks,
            verbose=1
        )

        # Salvar histórico
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history.history, f, indent=2)

        print(
            f"\nTreinamento concluído! Modelos salvos em: {self.output_dir}")

    def run(self):
        """Executa pipeline completa"""
        self.prepare_data()
        self.build_model()
        self.compile_model()
        self.setup_callbacks()
        self.train()

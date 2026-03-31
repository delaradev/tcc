"""Construção de datasets a partir de tiles"""
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional
import random
import numpy as np


class CPICDataset:
    """Classe para gerenciar dataset de pivôs centrais"""

    def __init__(self, dataset_path: str, img_size: int = 512):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size

    def load_pairs(self, split: str) -> List[Tuple[str, str]]:
        """
        Carrega pares imagem-máscara para um split

        Args:
            split: 'train' ou 'val'

        Returns:
            Lista de tuplas (caminho_imagem, caminho_mascara)
        """
        img_dir = self.dataset_path / f'{split}_images'
        msk_dir = self.dataset_path / f'{split}_masks'

        if not img_dir.exists() or not msk_dir.exists():
            raise FileNotFoundError(
                f"Diretórios não encontrados: {img_dir}, {msk_dir}")

        images = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])

        pairs = []
        for img_path in images:
            msk_path = msk_dir / f"{img_path.stem}{img_path.suffix}"
            if msk_path.exists():
                pairs.append((str(img_path), str(msk_path)))

        return pairs

    def create_dataset(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int,
        training: bool = True,
        randommix: bool = True,
        randommix_prob: float = 1.0
    ) -> tf.data.Dataset:
        """Cria dataset TensorFlow com augmentations"""

        img_paths = tf.constant([p[0] for p in pairs])
        msk_paths = tf.constant([p[1] for p in pairs])

        ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))

        if training:
            ds = ds.shuffle(1024)

        ds = ds.map(self._load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if training and randommix:
            ds = self._apply_randommix(ds, randommix_prob)
        else:
            ds = ds.map(lambda img, msk, _: (img, msk))

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def _load_pair(self, img_path: tf.Tensor, msk_path: tf.Tensor):
        """Carrega e pré-processa um par imagem-máscara"""
        # Carregar imagem
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_image(
            img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 255.0

        # Carregar máscara
        msk_bytes = tf.io.read_file(msk_path)
        msk = tf.image.decode_image(
            msk_bytes, channels=1, expand_animations=False)
        msk.set_shape([None, None, 1])
        msk = tf.image.resize(
            msk, [self.img_size, self.img_size], method='nearest')
        msk = tf.cast(msk, tf.float32) / 255.0
        msk = tf.where(msk >= 0.5, 1.0, 0.0)

        has_pos = tf.reduce_any(msk > 0.5)

        return img, msk, has_pos

    def _apply_randommix(self, ds, prob: float):
        """Aplica RandomMix conforme artigo"""
        # Separar positivos e negativos
        pos_ds = ds.filter(lambda img, msk, has_pos: has_pos)
        neg_ds = ds.filter(lambda img, msk, has_pos: tf.logical_not(has_pos))

        # Repetir positivos
        pos_ds = pos_ds.repeat()
        neg_ds = neg_ds.shuffle(2048)

        # Combinar e aplicar mix
        mixed = tf.data.Dataset.zip((neg_ds, pos_ds))

        def mix(neg, pos):
            neg_img, neg_msk, _ = neg
            pos_img, pos_msk, _ = pos

            r = tf.random.uniform([])

            def do_mix():
                return self._randommix_paste(neg_img, neg_msk, pos_img, pos_msk)

            img, msk = tf.cond(r <= prob, do_mix, lambda: (neg_img, neg_msk))
            return img, msk

        mixed_neg = mixed.map(mix)

        # Manter positivos originais
        pos_clean = pos_ds.map(lambda img, msk, _: (img, msk))

        return pos_clean.concatenate(mixed_neg).shuffle(2048)

    def _randommix_paste(self, neg_img, neg_msk, pos_img, pos_msk):
        """RandomMix: cola quarter do positivo no negativo"""
        q = self.img_size // 2  # quarter size
        max_xy = self.img_size - q

        # Posição para colar
        x = tf.random.uniform([], minval=0, maxval=max_xy + 1, dtype=tf.int32)
        y = tf.random.uniform([], minval=0, maxval=max_xy + 1, dtype=tf.int32)

        # Quarter aleatório do positivo
        x2 = tf.random.uniform([], minval=0, maxval=q + 1, dtype=tf.int32)
        y2 = tf.random.uniform([], minval=0, maxval=q + 1, dtype=tf.int32)

        patch_img = pos_img[x2:x2+q, y2:y2+q, :]
        patch_msk = pos_msk[x2:x2+q, y2:y2+q, :]

        # Rotação aleatória
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        patch_img = tf.image.rot90(patch_img, k)
        patch_msk = tf.image.rot90(patch_msk, k)

        # Colar
        def paste(img, patch):
            top = img[:x, :, :]
            mid = img[x:x+q, :, :]
            bot = img[x+q:, :, :]

            mid_left = mid[:, :y, :]
            mid_right = mid[:, y+q:, :]
            mid_new = tf.concat([mid_left, patch, mid_right], axis=1)

            return tf.concat([top, mid_new, bot], axis=0)

        out_img = paste(neg_img, patch_img)
        out_msk = paste(neg_msk, patch_msk)
        out_msk = tf.where(out_msk >= 0.5, 1.0, 0.0)

        return out_img, out_msk

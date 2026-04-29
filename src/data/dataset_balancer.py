import tensorflow as tf
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

from src.utils.logging import get_logger

logger = get_logger(__name__)

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
MSK_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_matching_image(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_mask_grayscale(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    return m


def foreground_ratio(mask: np.ndarray) -> float:
    if mask.max() <= 0:
        return 0.0
    m = mask.astype(np.float32) / 255.0
    return float(np.sum(m > 0.5) / m.size)


def copy_pair(src_img: Path, src_msk: Path, dst_img: Path, dst_msk: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_msk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_msk, dst_msk)


def create_balanced_dataset(
    src_root: str,
    dst_root: str,
    min_fg_ratio: float = 0.003,
    desired_pos_ratio: float = 0.7,
    seed: int = 42,
    use_all_positives: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Creating balanced dataset from {src_root} to {dst_root}")
    logger.info(
        f"Min FG ratio: {min_fg_ratio}, desired pos ratio: {desired_pos_ratio}")

    src_path = Path(src_root)
    dst_path = Path(dst_root)

    img_dir = src_path / 'train_images'
    msk_dir = src_path / 'train_masks'

    dst_img_dir = dst_path / 'train_images'
    dst_msk_dir = dst_path / 'train_masks'
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_msk_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted([p for p in msk_dir.iterdir()
                        if p.suffix.lower() in MSK_EXTS])
    logger.info(f"Total masks: {len(mask_files)}")

    positives = []
    negatives = []
    skipped_no_img = 0
    skipped_bad_mask = 0

    for msk_path in mask_files:
        stem = msk_path.stem
        img_path = find_matching_image(img_dir, stem)
        if img_path is None:
            skipped_no_img += 1
            continue
        try:
            m = read_mask_grayscale(msk_path)
        except Exception:
            skipped_bad_mask += 1
            continue
        fg = foreground_ratio(m)
        if fg >= min_fg_ratio:
            positives.append((img_path, msk_path, fg))
        else:
            negatives.append((img_path, msk_path, fg))

    logger.info(f"Positives (fg >= {min_fg_ratio}): {len(positives)}")
    logger.info(f"Negatives: {len(negatives)}")
    if skipped_no_img:
        logger.warning(f"No matching image for {skipped_no_img} masks")
    if skipped_bad_mask:
        logger.warning(f"Invalid masks: {skipped_bad_mask}")

    if not positives:
        raise RuntimeError("No positive samples found in training set.")

    selected_pos = positives if use_all_positives else sorted(
        positives, key=lambda x: x[2], reverse=True)

    target_neg = int(round(len(selected_pos) *
                     (1.0 - desired_pos_ratio) / max(desired_pos_ratio, 1e-9)))
    target_neg = min(target_neg, len(negatives))
    selected_neg = random.sample(
        negatives, target_neg) if target_neg > 0 else []

    logger.info(
        f"Selected positives: {len(selected_pos)}, negatives: {len(selected_neg)}")
    logger.info(
        f"Final ratio: {len(selected_pos)/(len(selected_pos)+len(selected_neg)):.1%}")

    copied_train = 0
    for img_path, msk_path, _ in selected_pos + selected_neg:
        dst_img = dst_img_dir / f"{msk_path.stem}{img_path.suffix}"
        dst_msk = dst_msk_dir / f"{msk_path.stem}{msk_path.suffix}"
        copy_pair(img_path, msk_path, dst_img, dst_msk)
        copied_train += 1

    logger.info(f"Copied {copied_train} training pairs")

    valid_img_src = src_path / 'valid_data' / 'valid_images'
    valid_msk_src = src_path / 'valid_data' / 'valid_masks'
    valid_img_dst = dst_path / 'valid_images'
    valid_msk_dst = dst_path / 'valid_masks'

    if valid_img_src.exists() and valid_msk_src.exists():
        valid_img_dst.mkdir(parents=True, exist_ok=True)
        valid_msk_dst.mkdir(parents=True, exist_ok=True)

        copied_valid_img = 0
        for file in valid_img_src.iterdir():
            if file.suffix.lower() in IMG_EXTS:
                shutil.copy2(file, valid_img_dst / file.name)
                copied_valid_img += 1

        copied_valid_msk = 0
        for file in valid_msk_src.iterdir():
            if file.suffix.lower() in MSK_EXTS:
                shutil.copy2(file, valid_msk_dst / file.name)
                copied_valid_msk += 1

        logger.info(
            f"Copied validation images: {copied_valid_img}, masks: {copied_valid_msk}")
    else:
        logger.warning("Validation directories missing, skipping copy.")


class CPICDatasetBuilder:
    def __init__(self, dataset_path: str, image_size: int = 512, seed: int = 42, max_train_samples: int = None):
        self.max_train_samples = max_train_samples
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.seed = seed
        self.splits = {
            'train': ('train_images', 'train_masks'),
            'valid': ('valid_images', 'valid_masks'),
        }

    def load_pairs(self, split: str) -> List[Tuple[str, str]]:
        if split not in self.splits:
            raise ValueError(
                f"Split must be one of {list(self.splits.keys())}")
        img_dir_name, mask_dir_name = self.splits[split]
        img_dir = self.dataset_path / img_dir_name
        mask_dir = self.dataset_path / mask_dir_name
        pairs = []
        for mask_path in sorted(mask_dir.glob('*.png')):
            img_path = img_dir / mask_path.name
            if img_path.exists():
                pairs.append((str(img_path), str(mask_path)))

        if split == 'train' and self.max_train_samples is not None and len(pairs) > self.max_train_samples:
            rng = random.Random(self.seed)
            pairs = rng.sample(pairs, self.max_train_samples)

        return pairs

    def _load_and_preprocess(self, image_path, mask_path, training=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        mask = tf.cast(mask, tf.float32) / 255.0

        if training:
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)
        return image, mask

    def build_dataset(self, split: str, batch_size: int = 1, training: bool = False) -> tf.data.Dataset:
        pairs = self.load_pairs(split)
        if not pairs:
            raise ValueError(f"No image-mask pairs found for split '{split}'")
        ds = tf.data.Dataset.from_tensor_slices(pairs)
        if training:
            ds = ds.shuffle(buffer_size=len(pairs),
                            seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda img, msk: self._load_and_preprocess(img, msk, training=training),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

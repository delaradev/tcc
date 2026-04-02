"""
================================================================================
CPIC DATASET BUILDER FOR TENSORFLOW
================================================================================

Implements data loading, preprocessing, and augmentation pipeline for
Center Pivot Irrigation Cropland (CPIC) segmentation.

Key features:
    - Efficient TF.Data pipeline with prefetching
    - RandomMix augmentation as described in Liu et al. (2023)
    - Class imbalance handling via strategic positive/negative sampling
    - Configurable train/validation splits

Based on methodology from:
Liu et al. (2023) - ISPRS Journal of Photogrammetry and Remote Sensing
Section 3.2: CPIC Training Dataset Generation

Author: Emanuel de Lara Ruas
Version: 1.0.0
================================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# MODULE CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


# -----------------------------------------------------------------------------
# DATA AUGMENTATION UTILITIES
# -----------------------------------------------------------------------------

class RandomMixAugmentation:
    """
    Implements RandomMix augmentation strategy from Liu et al. (2023)

    This technique randomly pastes a quarter of a positive sample
    (containing CPIC) into a negative sample (background only),
    creating synthetic training examples that improve model robustness.

    Reference: Liu et al. (2023), Section 3.2, Algorithm 1
    """

    @staticmethod
    def paste_quarter(
        negative_image: tf.Tensor,
        negative_mask: tf.Tensor,
        positive_image: tf.Tensor,
        positive_mask: tf.Tensor,
        image_size: int,
        seed: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Paste a random quarter of positive sample into negative sample

        Args:
            negative_image: Background image (no CPIC)
            negative_mask: Background mask (all zeros)
            positive_image: Image containing CPIC
            positive_mask: Mask with CPIC annotations
            image_size: Size of input images (assumed square)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        quarter_size = image_size // 2
        max_position = image_size - quarter_size

        # Random position in negative sample for pasting
        if seed is not None:
            tf.random.set_seed(seed)

        x_pos = tf.random.uniform(
            [], minval=0, maxval=max_position + 1, dtype=tf.int32
        )
        y_pos = tf.random.uniform(
            [], minval=0, maxval=max_position + 1, dtype=tf.int32
        )

        # Random quarter from positive sample (not just fixed quadrants)
        x_quarter = tf.random.uniform(
            [], minval=0, maxval=quarter_size + 1, dtype=tf.int32
        )
        y_quarter = tf.random.uniform(
            [], minval=0, maxval=quarter_size + 1, dtype=tf.int32
        )

        # Extract quarter patch
        patch_image = positive_image[
            x_quarter:x_quarter + quarter_size,
            y_quarter:y_quarter + quarter_size, :
        ]
        patch_mask = positive_mask[
            x_quarter:x_quarter + quarter_size,
            y_quarter:y_quarter + quarter_size, :
        ]

        # Random rotation (0°, 90°, 180°, 270°)
        rotation_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        patch_image = tf.image.rot90(patch_image, k=rotation_k)
        patch_mask = tf.image.rot90(patch_mask, k=rotation_k)

        # Paste patch into negative sample
        def paste_into(target: tf.Tensor, patch: tf.Tensor) -> tf.Tensor:
            """Helper to paste patch into target image"""
            top = target[:x_pos, :, :]
            middle = target[x_pos:x_pos + quarter_size, :, :]
            bottom = target[x_pos + quarter_size:, :, :]

            left = middle[:, :y_pos, :]
            right = middle[:, y_pos + quarter_size:, :]
            middle_new = tf.concat([left, patch, right], axis=1)

            return tf.concat([top, middle_new, bottom], axis=0)

        augmented_image = paste_into(negative_image, patch_image)
        augmented_mask = paste_into(negative_mask, patch_mask)
        augmented_mask = tf.where(augmented_mask >= 0.5, 1.0, 0.0)

        return augmented_image, augmented_mask


# -----------------------------------------------------------------------------
# DATASET BUILDER
# -----------------------------------------------------------------------------

class CPICDatasetBuilder:
    """
    Builds TensorFlow datasets for CPIC segmentation training

    Handles:
        - Loading image-mask pairs from directory structure
        - Data augmentation (flipping, RandomMix)
        - Class balancing via strategic sampling
        - Efficient pipelining with prefetching

    Expected directory structure:
        dataset_path/
            train_images/
                tile_00001.png
                tile_00002.png
                ...
            train_masks/
                tile_00001.png
                tile_00002.png
                ...
            valid_images/
                ...
            valid_masks/
                ...
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        image_size: int = 512,
        seed: int = 42
    ):
        """
        Initialize dataset builder

        Args:
            dataset_path: Root directory containing train/valid subdirectories
            image_size: Target image size (square, typically 512)
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.seed = seed

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}"
            )

        logger.info(f"Initialized CPICDatasetBuilder with:")
        logger.info(f"  Path: {self.dataset_path}")
        logger.info(f"  Image size: {image_size}x{image_size}")
        logger.info(f"  Seed: {seed}")

    def _get_split_dirs(self, split: str) -> Tuple[Path, Path]:
        """
        Get image and mask directories for a given split

        Args:
            split: 'train' or 'valid'

        Returns:
            Tuple of (image_dir, mask_dir)
        """
        image_dir = self.dataset_path / f"{split}_images"
        mask_dir = self.dataset_path / f"{split}_masks"

        return image_dir, mask_dir

    def load_pairs(self, split: str) -> List[Tuple[str, str]]:
        """
        Load all image-mask pairs for a given split

        Args:
            split: 'train' or 'valid'

        Returns:
            List of (image_path, mask_path) tuples

        Raises:
            FileNotFoundError: If directories don't exist
        """
        image_dir, mask_dir = self._get_split_dirs(split)

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Directories not found:\n"
                f"  Images: {image_dir}\n"
                f"  Masks: {mask_dir}"
            )

        # Find all image files
        image_files = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        pairs = []
        missing_masks = 0

        for img_path in image_files:
            # Look for corresponding mask with same stem
            mask_path = mask_dir / f"{img_path.stem}{img_path.suffix}"

            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
            else:
                missing_masks += 1

        if missing_masks > 0:
            logger.warning(
                f"Found {missing_masks} images without corresponding masks "
                f"in {split} split"
            )

        logger.info(
            f"Loaded {len(pairs)} pairs from {split} split"
        )

        return pairs

    def _load_image(self, path: tf.Tensor) -> tf.Tensor:
        """Load and preprocess a single image"""
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255.0

        return image

    def _load_mask(self, path: tf.Tensor) -> tf.Tensor:
        """Load and preprocess a single mask"""
        mask_bytes = tf.io.read_file(path)
        mask = tf.image.decode_image(
            mask_bytes,
            channels=1,
            expand_animations=False
        )
        mask.set_shape([None, None, 1])
        mask = tf.image.resize(
            mask,
            [self.image_size, self.image_size],
            method='nearest'
        )
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.where(mask >= 0.5, 1.0, 0.0)

        return mask

    def _load_pair_with_label(
        self,
        image_path: tf.Tensor,
        mask_path: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Load image-mask pair and determine if it contains CPIC

        Returns:
            Tuple of (image, mask, has_cpic_flag)
        """
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        has_cpic = tf.reduce_any(mask > 0.5)

        return image, mask, has_cpic

    def _apply_basic_augmentation(
        self,
        image: tf.Tensor,
        mask: tf.Tensor,
        _: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply basic augmentations (horizontal/vertical flips)

        Returns:
            Tuple of (augmented_image, augmented_mask, has_cpic_flag)
        """
        seed = tf.random.uniform(
            [], minval=0, maxval=2**31 - 1, dtype=tf.int32)

        # Horizontal flip
        flip_h = tf.random.stateless_uniform([], seed=[seed, 0])
        image = tf.cond(
            flip_h > 0.5,
            lambda: tf.image.flip_left_right(image),
            lambda: image
        )
        mask = tf.cond(
            flip_h > 0.5,
            lambda: tf.image.flip_left_right(mask),
            lambda: mask
        )

        # Vertical flip
        flip_v = tf.random.stateless_uniform([], seed=[seed, 1])
        image = tf.cond(
            flip_v > 0.5,
            lambda: tf.image.flip_up_down(image),
            lambda: image
        )
        mask = tf.cond(
            flip_v > 0.5,
            lambda: tf.image.flip_up_down(mask),
            lambda: mask
        )

        return image, mask, _

    def _apply_randommix(
        self,
        dataset: tf.data.Dataset,
        probability: float = 1.0
    ) -> tf.data.Dataset:
        """
        Apply RandomMix augmentation to balance positive/negative samples

        Args:
            dataset: Input dataset with (image, mask, has_cpic) tuples
            probability: Probability of applying RandomMix to negative samples

        Returns:
            Augmented dataset with balanced positive/negative distribution
        """
        # Separate positive and negative samples
        positive_samples = dataset.filter(
            lambda img, msk, has_pos: has_pos
        )
        negative_samples = dataset.filter(
            lambda img, msk, has_pos: tf.logical_not(has_pos)
        )

        # Repeat positive samples to ensure availability
        positive_samples = positive_samples.repeat()
        negative_samples = negative_samples.shuffle(2048, seed=self.seed)

        # Combine for mixing
        mixed_pairs = tf.data.Dataset.zip((negative_samples, positive_samples))

        def mix_samples(negative, positive):
            neg_img, neg_msk, _ = negative
            pos_img, pos_msk, _ = positive

            apply_mix = tf.random.uniform([]) <= probability

            def do_mix():
                return RandomMixAugmentation.paste_quarter(
                    neg_img, neg_msk, pos_img, pos_msk,
                    self.image_size, self.seed
                )

            def no_mix():
                return neg_img, neg_msk

            return tf.cond(apply_mix, do_mix, no_mix)

        mixed_negatives = mixed_pairs.map(mix_samples)

        # Keep original positives unchanged
        positive_clean = positive_samples.map(
            lambda img, msk, _: (img, msk)
        )

        # Combine and shuffle
        balanced_dataset = positive_clean.concatenate(mixed_negatives)
        balanced_dataset = balanced_dataset.shuffle(
            2048, seed=self.seed, reshuffle_each_iteration=True
        )

        return balanced_dataset

    def build_dataset(
        self,
        split: str,
        batch_size: int,
        training: bool = True,
        use_randommix: bool = True,
        randommix_probability: float = 1.0,
        cache: bool = False
    ) -> tf.data.Dataset:
        """
        Build a TensorFlow dataset for training or validation

        Args:
            split: 'train' or 'valid'
            batch_size: Number of samples per batch
            training: Whether to apply training augmentations
            use_randommix: Whether to apply RandomMix augmentation
            randommix_probability: Probability of applying RandomMix
            cache: Whether to cache dataset (useful for small datasets)

        Returns:
            Configured tf.data.Dataset
        """
        pairs = self.load_pairs(split)

        if not pairs:
            raise ValueError(f"No valid pairs found for split: {split}")

        # Create TensorFlow dataset from paths
        image_paths = tf.constant([p[0] for p in pairs])
        mask_paths = tf.constant([p[1] for p in pairs])
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

        if training:
            dataset = dataset.shuffle(
                buffer_size=min(2048, len(pairs)),
                seed=self.seed,
                reshuffle_each_iteration=True
            )

        # Load images and masks
        dataset = dataset.map(
            self._load_pair_with_label,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply augmentations (training only)
        if training:
            dataset = dataset.map(
                self._apply_basic_augmentation,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Optional caching
        if cache:
            dataset = dataset.cache()

        # Apply RandomMix for training
        if training and use_randommix:
            dataset = self._apply_randommix(dataset, randommix_probability)
        else:
            dataset = dataset.map(
                lambda img, msk, _: (img, msk),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch and prefetch
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.info(
            f"Built {split} dataset: "
            f"{len(pairs)} samples, batch_size={batch_size}"
        )

        return dataset

    def get_dataset_statistics(self, split: str) -> Dict[str, float]:
        """
        Compute statistics about a dataset split

        Args:
            split: 'train' or 'valid'

        Returns:
            Dictionary with statistics
        """
        pairs = self.load_pairs(split)

        if not pairs:
            return {}

        # Sample a few images to compute statistics
        sample_size = min(50, len(pairs))
        selected_pairs = pairs[:sample_size]

        positive_count = 0
        total_pixels = 0
        cpic_pixels = 0

        for img_path, msk_path in selected_pairs:
            mask = tf.io.read_file(msk_path)
            mask = tf.image.decode_image(mask, channels=1)
            mask = tf.cast(mask, tf.float32) / 255.0
            mask = tf.where(mask >= 0.5, 1.0, 0.0)

            if tf.reduce_any(mask > 0.5):
                positive_count += 1

            total_pixels += tf.size(mask).numpy()
            cpic_pixels += tf.reduce_sum(mask).numpy()

        stats = {
            'total_pairs': len(pairs),
            'positive_ratio': positive_count / sample_size if sample_size > 0 else 0,
            'cpic_pixel_ratio': cpic_pixels / total_pixels if total_pixels > 0 else 0,
            'samples_analyzed': sample_size
        }

        logger.info(f"Statistics for {split} split:")
        logger.info(f"  Total pairs: {stats['total_pairs']}")
        logger.info(f"  Positive samples ratio: {stats['positive_ratio']:.2%}")
        logger.info(f"  CPIC pixel ratio: {stats['cpic_pixel_ratio']:.4%}")

        return stats


# -----------------------------------------------------------------------------
# CLI INTERFACE (OPTIONAL)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick test/demo of dataset builder functionality
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test CPIC dataset builder")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset_balanced",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for testing"
    )

    args = parser.parse_args()

    # Initialize builder
    builder = CPICDatasetBuilder(args.dataset_path)

    # Print statistics
    try:
        builder.get_dataset_statistics('train')
        builder.get_dataset_statistics('valid')
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.info(
            "Please ensure dataset is prepared with train/valid structure")
        sys.exit(1)

    # Build and iterate through a small dataset
    try:
        dataset = builder.build_dataset(
            split='train',
            batch_size=args.batch_size,
            training=True
        )

        logger.info("\nTesting dataset iteration...")
        for i, (images, masks) in enumerate(dataset.take(3)):
            logger.info(
                f"Batch {i+1}: images shape={images.shape}, masks shape={masks.shape}")

        logger.info("\nDataset builder test completed successfully")

    except Exception as e:
        logger.error(f"Error building dataset: {e}")
        sys.exit(1)

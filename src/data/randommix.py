import random
import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image


class RandomMixAugmentation:
    @staticmethod
    def apply_randommix(
        img_neg: np.ndarray,
        mask_neg: np.ndarray,
        img_pos: np.ndarray,
        mask_pos: np.ndarray,
        tile_size: int = 512,
        crop_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = tile_size, tile_size
        y = np.random.randint(0, h - crop_size)
        x = np.random.randint(0, w - crop_size)
        quad_idx = np.random.randint(0, 4)

        if quad_idx == 0:
            pos_crop = img_pos[:crop_size, :crop_size, :]
            pos_mask_crop = mask_pos[:crop_size, :crop_size, :]
        elif quad_idx == 1:
            pos_crop = img_pos[:crop_size, -crop_size:, :]
            pos_mask_crop = mask_pos[:crop_size, -crop_size:, :]
        elif quad_idx == 2:
            pos_crop = img_pos[-crop_size:, :crop_size, :]
            pos_mask_crop = mask_pos[-crop_size:, :crop_size, :]
        else:
            pos_crop = img_pos[-crop_size:, -crop_size:, :]
            pos_mask_crop = mask_pos[-crop_size:, -crop_size:, :]

        k = np.random.randint(0, 4)
        pos_crop = np.rot90(pos_crop, k=k)
        pos_mask_crop = np.rot90(pos_mask_crop, k=k)

        new_img = img_neg.copy()
        new_mask = mask_neg.copy()
        new_img[y:y + crop_size, x:x + crop_size, :] = pos_crop
        new_mask[y:y + crop_size, x:x + crop_size, :] = pos_mask_crop

        return new_img, new_mask


def generate_randommix_dataset(
    original_train_path: str = "data/dataset",
    output_path: str = "data/dataset_randommix",
    min_fg_ratio: float = 0.003,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    orig_img_dir = Path(original_train_path) / "train_images"
    orig_msk_dir = Path(original_train_path) / "train_masks"

    mask_files = sorted(orig_msk_dir.glob("*.png"))
    positives = []
    negatives = []

    for msk_path in mask_files:
        img_path = orig_img_dir / msk_path.name
        if not img_path.exists():
            continue
        mask = np.array(Image.open(msk_path))
        fg = np.sum(mask > 128) / mask.size
        if fg >= min_fg_ratio:
            positives.append((img_path, msk_path, fg))
        elif fg == 0.0:
            negatives.append((img_path, msk_path))

    if not positives or not negatives:
        raise RuntimeError(
            f"Cannot apply RandomMix: need at least one positive and one pure negative tile. "
            f"Positives: {len(positives)}, Negatives: {len(negatives)}"
        )

    out_img_dir = Path(output_path) / "train_images"
    out_msk_dir = Path(output_path) / "train_masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    for idx, (neg_img_path, neg_mask_path) in enumerate(negatives):
        neg_img = np.array(Image.open(neg_img_path))
        neg_mask = np.array(Image.open(neg_mask_path))[..., np.newaxis]

        pos_idx = random.randint(0, len(positives) - 1)
        pos_img_path, pos_mask_path, _ = positives[pos_idx]
        pos_img = np.array(Image.open(pos_img_path))
        pos_mask = np.array(Image.open(pos_mask_path))[..., np.newaxis]

        new_img, new_mask = RandomMixAugmentation.apply_randommix(
            neg_img, neg_mask, pos_img, pos_mask
        )

        new_name = f"randommix_{idx:05d}.png"
        Image.fromarray(new_img.astype(np.uint8)).save(out_img_dir / new_name)
        Image.fromarray(new_mask.squeeze().astype(
            np.uint8)).save(out_msk_dir / new_name)

    print(
        f"RandomMix dataset generated: {len(negatives)} samples at {output_path}")


if __name__ == "__main__":
    generate_randommix_dataset()

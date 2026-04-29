import tensorflow as tf
from pathlib import Path
import numpy as np


class PredictionSaver(tf.keras.callbacks.Callback):
    def __init__(self, validation_ds, save_dir: str, max_samples: int = 4, threshold: float = 0.5):
        super().__init__()
        self.validation_ds = validation_ds
        self.save_dir = Path(save_dir)
        self.max_samples = max_samples
        self.threshold = threshold
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        epoch_dir = self.save_dir / f'epoch_{epoch+1:03d}'
        epoch_dir.mkdir(exist_ok=True)

        for i, (images, masks) in enumerate(self.validation_ds.take(1)):
            predictions = self.model.predict(images, verbose=0)

            for j in range(min(self.max_samples, len(images))):
                self._save_image(images[j], epoch_dir /
                                 f'sample_{j}_image.png')
                self._save_mask(masks[j], epoch_dir /
                                f'sample_{j}_true_mask.png')
                pred_mask = (predictions[j] >=
                             self.threshold).astype(np.float32)
                self._save_mask(pred_mask, epoch_dir /
                                f'sample_{j}_pred_mask.png')

    def _save_image(self, image, path):
        from PIL import Image
        if hasattr(image, 'numpy'):
            img = (image.numpy() * 255).astype(np.uint8)
        else:
            img = (image * 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    def _save_mask(self, mask, path):
        from PIL import Image
        if hasattr(mask, 'numpy'):
            msk = mask.numpy()
        else:
            msk = mask
        if msk.ndim == 3 and msk.shape[-1] == 1:
            msk = msk.squeeze(axis=-1)
        elif msk.ndim == 3 and msk.shape[0] == 1:
            msk = msk.squeeze(axis=0)
        msk = (msk * 255).astype(np.uint8)
        Image.fromarray(msk).save(path)


class GPUMemoryMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            mem_used = result.stdout.strip()
            print(f"GPU Memory: {mem_used}")
        except:
            pass

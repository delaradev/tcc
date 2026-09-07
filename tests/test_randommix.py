"""Regressão para o bug crítico corrigido em src/data/randommix.py: a classificação
de "negativo" usava `fg == 0.0` (exatamente zero), inconsistente com o critério de
create_balanced_dataset() (`fg < min_fg_ratio`). Qualquer tile com um pouco de SIPC
abaixo do limiar (0 < fg < min_fg_ratio) era descartado em silêncio — nem virava
positivo, nem negativo, nem entrava no dataset final. Não depende de TensorFlow."""
import numpy as np
from PIL import Image

from src.data.randommix import RandomMixAugmentation, generate_randommix_dataset


def _make_pair(img_dir, mask_dir, name, mask):
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(img_dir / name)
    Image.fromarray(mask).save(mask_dir / name)


def test_borderline_foreground_counts_as_negative(tmp_path):
    img_dir = tmp_path / 'train_images'
    mask_dir = tmp_path / 'train_masks'
    img_dir.mkdir()
    mask_dir.mkdir()

    min_fg_ratio = 0.01

    # positivo claro (fg bem acima do limiar)
    positive_mask = np.zeros((64, 64), dtype=np.uint8)
    positive_mask[10:50, 10:50] = 255
    _make_pair(img_dir, mask_dir, 'tile_pos.png', positive_mask)

    # negativo puro (fg == 0)
    _make_pair(img_dir, mask_dir, 'tile_neg_pure.png',
               np.zeros((64, 64), dtype=np.uint8))

    # fronteira: fg > 0 mas < min_fg_ratio — este é o caso que o bug descartava
    borderline_mask = np.zeros((64, 64), dtype=np.uint8)
    borderline_mask[0:2, 0:2] = 255  # 4/4096 ≈ 0.00098 < 0.01
    _make_pair(img_dir, mask_dir, 'tile_borderline.png', borderline_mask)

    output_path = tmp_path / 'randommix_out'
    generate_randommix_dataset(
        original_train_path=str(tmp_path),
        output_path=str(output_path),
        min_fg_ratio=min_fg_ratio,
        prob=1.0,
        crop_size=32,
        seed=42,
    )

    out_files = sorted((output_path / 'train_images').glob('*.png'))
    # Devem existir 2 negativos processados: o puro e o de fronteira. Antes da
    # correção, apenas 1 (o puro) seria contabilizado e o de fronteira desaparecia.
    assert len(out_files) == 2


def test_apply_randommix_rotation_is_never_identity():
    """O Algorithm 1 do artigo especifica rotação de 90/180/270 (nunca 0/identidade)
    na amostra positiva antes de colar. Usa um padrão assimétrico (gradiente) para
    que a rotação seja detectável: com crop_size == tile_size, o bloco colado inteiro
    deve ser sempre uma das 3 rotações não-triviais de img_pos, nunca o próprio img_pos."""
    size = 64
    gradient = np.tile(np.arange(size, dtype=np.uint8), (size, 1))
    img_pos = np.stack([gradient, gradient, gradient], axis=-1)
    mask_pos = gradient[:, :, np.newaxis]

    img_neg = np.zeros((size, size, 3), dtype=np.uint8)
    mask_neg = np.zeros((size, size, 1), dtype=np.uint8)

    identity = img_pos
    valid_rotations = [np.rot90(img_pos, k=k) for k in (1, 2, 3)]

    for _ in range(30):
        new_img, _ = RandomMixAugmentation.apply_randommix(
            img_neg, mask_neg, img_pos, mask_pos, crop_size=size)
        assert not np.array_equal(new_img, identity)
        assert any(np.array_equal(new_img, rotated) for rotated in valid_rotations)

# scripts/apply_randommix.py
from src.data.dataset_balancer import (
    CPICDatasetBuilder, RandomMixAugmentation,
    foreground_ratio, read_mask_grayscale, IMG_EXTS, MSK_EXTS
)
from PIL import Image
import shutil
import random
import numpy as np
import sys
from pathlib import Path
# adiciona 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_randommix_dataset(
    original_train_path="data/dataset",
    balanced_train_path="data/dataset_balanced",
    output_path="data/dataset_mixed",
    min_fg_ratio=0.003,
    seed=42
):
    """
    Gera dataset com RandomMix a partir dos dados originais, combinado com
    o treino já balanceado (opcional) + validação original.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Carregar os pares de treino originais (não balanceados)
    orig_img_dir = Path(original_train_path) / "train_images"
    orig_msk_dir = Path(original_train_path) / "train_masks"

    print("Carregando pares de treino originais...")
    orig_pairs = []
    for msk_path in sorted(orig_msk_dir.glob("*.png")):
        stem = msk_path.stem
        img_path = orig_img_dir / f"{stem}.png"  # assume PNG
        if img_path.exists():
            orig_pairs.append((img_path, msk_path))

    # 2. Calcular foreground_ratio para cada par
    print("Calculando foreground ratios...")
    positives = []
    negatives = []
    for img_p, msk_p in orig_pairs:
        try:
            mask = read_mask_grayscale(msk_p)
        except Exception:
            continue
        fg = foreground_ratio(mask)
        if fg >= min_fg_ratio:
            positives.append((img_p, msk_p, fg))
        elif fg == 0.0:
            negatives.append((img_p, msk_p))
        # tiles com foreground > 0 porém < min_fg_ratio são ignorados (não são puros nem positivos para RandomMix)

    print(f"Positivos (fg >= {min_fg_ratio}): {len(positives)}")
    print(f"Negativos puros (fg = 0): {len(negatives)}")

    if not positives or not negatives:
        print(
            "❌ Não foi possível aplicar RandomMix: faltam positivos e/ou negativos puros.")
        if not negatives:
            print("   Nenhum tile 100% sem pivôs encontrado. RandomMix inviável.")
        return

    # 3. Preparar diretórios de saída
    out_img_dir = Path(output_path) / "train_images"
    out_msk_dir = Path(output_path) / "train_masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copiar todos os pares do dataset balanceado para o novo dataset
    balanced_img_dir = Path(balanced_train_path) / "train_images"
    balanced_msk_dir = Path(balanced_train_path) / "train_masks"
    print("Copiando dataset balanceado...")
    copied_balance = 0
    for img_file in sorted(balanced_img_dir.glob("*.png")):
        msk_file = balanced_msk_dir / img_file.name
        if msk_file.exists():
            shutil.copy2(img_file, out_img_dir / img_file.name)
            shutil.copy2(msk_file, out_msk_dir / img_file.name)
            copied_balance += 1
    print(f"{copied_balance} pares do balanceado copiados.")

    # 5. Gerar amostras RandomMix
    print("Gerando RandomMix...")
    num_mix = len(negatives)  # uma amostra por negativo, conforme Algoritmo 1
    for idx, (neg_img_path, neg_mask_path) in enumerate(negatives):
        # Carrega negativo (já sabemos que é puro, foreground=0)
        neg_img = np.array(Image.open(neg_img_path))
        neg_mask = np.array(Image.open(neg_mask_path)
                            )[..., np.newaxis]  # (H,W,1)

        # Seleciona positivo aleatório
        pos_img_path, pos_mask_path, _ = random.choice(positives)
        pos_img = np.array(Image.open(pos_img_path))
        pos_mask = np.array(Image.open(pos_mask_path))[..., np.newaxis]

        new_img, new_mask = RandomMixAugmentation.apply_randommix(
            neg_img, neg_mask, pos_img, pos_mask
        )

        new_name = f"randommix_{idx:05d}.png"
        Image.fromarray(new_img.astype(np.uint8)).save(out_img_dir / new_name)
        Image.fromarray(new_mask.squeeze().astype(
            np.uint8)).save(out_msk_dir / new_name)

        if (idx + 1) % 500 == 0:
            print(f"   {idx+1}/{num_mix} amostras geradas")

    print(f"✅ {num_mix} amostras RandomMix geradas e adicionadas ao treino.")

    # 6. Copiar validação original
    print("Copiando validação original...")
    valid_img_src = Path(original_train_path) / "valid_data" / "valid_images"
    valid_msk_src = Path(original_train_path) / "valid_data" / "valid_masks"
    valid_img_dst = Path(output_path) / "valid_images"
    valid_msk_dst = Path(output_path) / "valid_masks"
    if valid_img_src.exists() and valid_msk_src.exists():
        valid_img_dst.mkdir(exist_ok=True)
        valid_msk_dst.mkdir(exist_ok=True)
        for f in valid_img_src.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                shutil.copy2(f, valid_img_dst / f.name)
        for f in valid_msk_src.iterdir():
            if f.suffix.lower() in MSK_EXTS:
                shutil.copy2(f, valid_msk_dst / f.name)
        print(
            f"   Validação copiada: {len(list(valid_img_dst.iterdir()))} imagens")
    else:
        print("   Diretórios de validação não encontrados.")

    print(f"\n🚀 Dataset misto criado em {output_path}")
    print(f"   Treino: {len(list(out_img_dir.iterdir()))} amostras")
    print(f"   Configurar balanced_path no config.yaml como '{output_path}'")


if __name__ == "__main__":
    generate_randommix_dataset()

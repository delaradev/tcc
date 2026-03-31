"""
Gera tiles de 512x512 com normalização adequada para o modelo
"""
import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional


# =========================
# CONFIGURAÇÕES
# =========================
tif_path = "ibge/landsat_cpic_2023.tif"  # ← usar o exportado do GEE
out_dir = Path("tiles/images")
tile_size = 512
overlap = 32  # Overlap para evitar bordas (opcional)


out_dir.mkdir(parents=True, exist_ok=True)


def normalize_band_percentile(band: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """
    Normaliza uma banda usando percentis para remover outliers
    Retorna array normalizado [0, 1]
    """
    # Ignorar valores nulos ou extremos
    valid = band[~np.isnan(band)]
    if len(valid) == 0:
        return np.zeros_like(band)
    
    low_val = np.percentile(valid, p_low)
    high_val = np.percentile(valid, p_high)
    
    if high_val <= low_val:
        return np.zeros_like(band)
    
    normalized = (band - low_val) / (high_val - low_val)
    return np.clip(normalized, 0, 1)


def normalize_stack(stack: np.ndarray, method: str = 'global') -> np.ndarray:
    """
    Normaliza stack de 3 bandas para [0, 255]
    
    Args:
        stack: array shape (H, W, 3) com valores float
        method: 'global' - percentis globais por banda
                'independent' - percentis independentes por banda
    
    Returns:
        array uint8 shape (H, W, 3)
    """
    normalized = np.zeros_like(stack, dtype=np.float32)
    
    for b in range(3):
        band = stack[:, :, b]
        
        if method == 'global':
            # Usar percentis fixos baseados no artigo
            if b == 0:  # EVI_max
                p_low, p_high = 0.0, 0.7
            elif b == 1:  # BSI_max
                p_low, p_high = -0.1, 0.3
            else:  # GREEN_median
                p_low, p_high = 0.0, 0.15
            
            band_norm = (band - p_low) / (p_high - p_low)
            band_norm = np.clip(band_norm, 0, 1)
            
        else:  # independent
            band_norm = normalize_band_percentile(band, p_low=2, p_high=98)
        
        normalized[:, :, b] = band_norm
    
    # Converter para uint8
    return (normalized * 255).astype(np.uint8)


def create_tiles_with_overlap(
    src_path: Path,
    out_dir: Path,
    tile_size: int = 512,
    overlap: int = 32,
    normalize_method: str = 'global'
) -> Tuple[int, int, int]:
    """
    Cria tiles com overlap para evitar bordas
    
    Returns:
        (total_tiles, width, height)
    """
    print(f"📂 Abrindo: {src_path}")
    
    with rasterio.open(src_path) as src:
        width = src.width
        height = src.height
        bands = src.count
        
        print(f"  Dimensões: {width} x {height} x {bands}")
        print(f"  Resolução: {src.res}")
        print(f"  CRS: {src.crs}")
        
        if bands != 3:
            print(f"⚠️ Aviso: {bands} bandas detectadas. Esperado 3.")
        
        # Calcular steps com overlap
        step = tile_size - overlap
        n_tiles_x = (width - tile_size + step - 1) // step + 1
        n_tiles_y = (height - tile_size + step - 1) // step + 1
        
        print(f"  Tiles: {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y}")
        print(f"  Step: {step}px, Overlap: {overlap}px")
        
        tile_id = 0
        
        for y_idx in range(n_tiles_y):
            y = min(y_idx * step, height - tile_size)
            
            for x_idx in range(n_tiles_x):
                x = min(x_idx * step, width - tile_size)
                
                # Ler janela
                window = Window(x, y, tile_size, tile_size)
                img = src.read(window=window)  # shape: (bands, H, W)
                
                # Reorganizar para (H, W, C)
                img = np.transpose(img, (1, 2, 0))
                
                # Normalizar
                img_norm = normalize_stack(img, method=normalize_method)
                
                # Salvar
                output_path = out_dir / f"tile_{tile_id:05d}.png"
                Image.fromarray(img_norm).save(output_path)
                
                tile_id += 1
                
                if tile_id % 100 == 0:
                    print(f"  Processados: {tile_id} tiles")
        
        print(f"\n✅ Total de tiles gerados: {tile_id}")
        return tile_id, width, height


def validate_tile_coverage(tiles_dir: Path, expected_count: int) -> bool:
    """Valida se todos os tiles foram gerados"""
    import glob
    
    tiles = sorted(glob.glob(str(tiles_dir / "tile_*.png")))
    actual_count = len(tiles)
    
    print(f"\n📊 Validação:")
    print(f"  Tiles esperados: {expected_count}")
    print(f"  Tiles gerados: {actual_count}")
    
    if actual_count == expected_count:
        print("  ✅ Todos os tiles foram gerados com sucesso!")
        return True
    else:
        print(f"  ⚠️ Diferença de {expected_count - actual_count} tiles")
        return False


def get_stats_from_tiles(tiles_dir: Path, n_samples: int = 100) -> dict:
    """Analisa estatísticas dos tiles gerados (para debug)"""
    import glob
    
    tiles = sorted(glob.glob(str(tiles_dir / "tile_*.png")))[:n_samples]
    
    if not tiles:
        return {}
    
    all_means = []
    
    for tile_path in tiles:
        img = Image.open(tile_path)
        arr = np.array(img) / 255.0
        all_means.append(arr.mean(axis=(0, 1)))
    
    all_means = np.array(all_means)
    
    stats = {
        'band_means': all_means.mean(axis=0),
        'band_stds': all_means.std(axis=0),
    }
    
    print(f"\n📊 Estatísticas dos tiles (amostra de {n_samples}):")
    for b, name in enumerate(['EVI_max', 'BSI_max', 'GREEN_median']):
        print(f"  {name}: mean={stats['band_means'][b]:.3f}, std={stats['band_stds'][b]:.3f}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerar tiles de imagens para treinamento")
    parser.add_argument("--tif_path", type=str, default="ibge/landsat_cpic_2023.tif",
                        help="Caminho do arquivo TIFF de entrada")
    parser.add_argument("--out_dir", type=str, default="tiles/images",
                        help="Diretório de saída para os tiles")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tamanho do tile (padrão: 512)")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Overlap entre tiles (padrão: 32)")
    parser.add_argument("--normalize", type=str, default="global",
                        choices=["global", "percentile"],
                        help="Método de normalização")
    parser.add_argument("--validate", action="store_true",
                        help="Validar estatísticas após geração")
    
    args = parser.parse_args()
    
    tif_path = Path(args.tif_path)
    out_dir = Path(args.out_dir)
    
    if not tif_path.exists():
        print(f"❌ Arquivo não encontrado: {tif_path}")
        exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_tiles, width, height = create_tiles_with_overlap(
        src_path=tif_path,
        out_dir=out_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        normalize_method=args.normalize
    )
    
    if args.validate:
        validate_tile_coverage(out_dir, n_tiles)
        get_stats_from_tiles(out_dir)
    
    print("\n✅ Processo concluído!")
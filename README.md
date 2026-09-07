# CPIC Brasil — Detecção de Pivôs Centrais com Deep Learning

Pipeline para identificação de Sistemas de Irrigação por Pivô Central (SIPC) em imagens
de satélite Landsat via U-Net, reproduzindo a metodologia de **Liu et al. (2023)** e
avaliando a generalização geográfica do modelo na região da **AMAJA** (Associação dos
Municípios do Alto Jacuí, RS) — objeto do TCC deste repositório.

- Artigo-base: Liu, X. et al. (2023). *Mapping annual center-pivot irrigated cropland
  in Brazil during the 1985–2021 period with cloud platforms and deep learning*. ISPRS
  Journal of Photogrammetry and Remote Sensing, 205, 227–245.
  <https://doi.org/10.1016/j.isprsjprs.2023.10.007>
- Dataset de treinamento: <https://zenodo.org/records/10046320>

---

## Índice

- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Pipeline 1 — Reproduzir Liu et al. (treino/teste)](#pipeline-1--reproduzir-liu-et-al-treinoteste)
- [Pipeline 2 — Construir a base da AMAJA e validar generalização](#pipeline-2--construir-a-base-da-amaja-e-validar-generalização)
- [Configuração](#configuração)
- [Testes](#testes)
- [Solução de problemas](#solução-de-problemas)

---

## Estrutura do projeto

```
tcc_code/
├── config/
│   ├── config.yaml            # Config principal (treino em cima do dataset de Liu et al.)
│   └── config_amaja.yaml      # Config para validar o modelo treinado na base da AMAJA
│
├── docs/                      # Artigo-base, TCC e material de apoio (PDFs)
│
├── notebooks/
│   └── cpic_training.py       # Driver do treino no Colab (exportado do notebook oficial)
│
├── src/
│   ├── main.py                 # CLI única: --mode train|predict|export|validate
│   ├── data/
│   │   ├── data_utils.py       # Extração/verificação do dataset Zenodo
│   │   ├── dataset_balancer.py # Balanceamento (70/30) + splits + tf.data.Dataset
│   │   ├── randommix.py        # Augmentação RandomMix (Algorithm 1 do artigo)
│   │   ├── tiles.py            # Recorte de GeoTIFF em tiles 512x512 (+ máscara pareada)
│   │   ├── amaja.py            # Municípios/AOI da AMAJA + download e filtro de pivôs ANA
│   │   ├── gee_export_amaja.py # Exportação Landsat via Google Earth Engine (roda no Colab)
│   │   └── review_tiles.py     # Apoio à validação humana das máscaras da AMAJA
│   ├── models/
│   │   ├── unet.py             # Arquitetura U-Net (Fig. 7 do artigo)
│   │   └── losses.py           # Tversky, Dice, combined loss
│   ├── training/
│   │   ├── train.py            # Trainer: prepara dados, treina, analisa pós-treino
│   │   ├── metrics.py          # IoU, Dice, Precision, Recall
│   │   └── callbacks.py        # Checkpoints, visualização por época
│   ├── inference/
│   │   └── predict.py          # Predição em arquivo/diretório + validação em split
│   ├── export/
│   │   └── model_exporter.py   # Exporta .keras -> SavedModel + TFLite
│   └── utils/
│       ├── gpu_utils.py        # Configuração de GPU/mixed precision
│       └── logging.py
│
├── tests/                      # pytest (testes sem TensorFlow rodam em qualquer ambiente)
│
├── requirements.txt             # Dependências principais
├── requirements-gee.txt         # Extra: earthengine-api/geemap (só p/ gee_export_amaja.py)
├── requirements-dev.txt         # Extra: pytest
├── pyproject.toml               # Config do pytest
│
├── data/                        # Dados (gerado localmente; fora do git)
├── runs/                        # Saída de cada treino (fora do git)
└── logs/                        # Logs (fora do git)
```

Os módulos em `src/data/` que não dependem de TensorFlow (`amaja.py`, `tiles.py`,
`review_tiles.py`, `gee_export_amaja.py`) podem ser importados e
executados isoladamente, sem precisar do TensorFlow instalado — útil para rodar a
etapa de preparo de dados da AMAJA num ambiente separado do treino.

---

## Instalação

Requer Python >= 3.10 (o venv local do projeto usa 3.12.9).

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate

pip install -r requirements.txt

# Só necessário para gerar a base da AMAJA via Google Earth Engine (ver Pipeline 2):
pip install -r requirements-gee.txt
```

Verificar GPU (opcional):
```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## Pipeline 1 — Reproduzir Liu et al. (treino/teste)

### 1. Obter o dataset
Baixe `dataset.zip` do Zenodo (link no topo deste README) e extraia em `data/dataset/`
(estrutura esperada: `train_images/`, `train_masks/`, `valid_images/`, `valid_masks/`).

### 2. Treinar
```bash
python src/main.py --mode train --config config/config.yaml
```
O `Trainer` monta automaticamente, na primeira execução:
- `data/dataset_balanced/` — dataset balanceado 70/30 positivo/negativo (`create_balanced_dataset`);
- `data/dataset_balanced_randommix/` — augmentação RandomMix, se `training.randommix: true`.

Do conjunto de treino resultante, **10% é separado como validação interna**
(`internal_val_fraction`) para orientar early stopping, checkpoint e redução de LR — o
split `valid_images/valid_masks` do Zenodo é o **conjunto de teste final**, mantido
isolado do treinamento e usado apenas na análise pós-treino
(`Trainer.post_training_analysis`, salva em `runs/<run>/analysis/`).

Cada execução grava em `runs/<run>/`:
- `best_model.keras` — melhor modelo por IoU na validação interna (o que é avaliado);
- `last_model.keras` — sobrescrito a cada época (não acumula um arquivo por época) para
  retomar (`--resume`) sem perder progresso e sem esgotar espaço em disco/Drive;
- `training_log.csv` — métricas por época, gravadas incrementalmente (não se perde ao
  interromper o treino, ao contrário de `history.json`, só escrito ao final do `fit()`).

Para retomar (detecta a época automaticamente pelo `training_log.csv`):
```bash
python src/main.py --mode train --config config/config.yaml \
    --resume runs/<run>/last_model.keras
```

### 3. Avaliar no conjunto de teste reservado
```bash
python src/main.py --mode validate --config config/config.yaml \
    --model runs/<run>/best_model.keras
```

### 4. Exportar o modelo (opcional)
```bash
python src/main.py --mode export --config config/config.yaml \
    --model runs/<run>/best_model.keras --output runs/<run>/exported
```

---

## Pipeline 2 — Construir a base da AMAJA e validar generalização

Reproduz a Seção 4.4 do TCC: avaliar o modelo treinado acima numa região geográfica
nunca vista (os 20 municípios da AMAJA/RS), com máscaras derivadas de dados oficiais
da ANA e validadas visualmente.

### 1–3. Municípios, AOI e pivôs da ANA (roda localmente, sem GPU/GEE)
```bash
python src/data/amaja.py --output_dir data/raw/amaja --ana_dir data/raw/ana
```
Baixa os limites dos 20 municípios (API de Malhas do IBGE) e o shapefile nacional de
pivôs da ANA, filtra para a AMAJA e salva em `data/raw/amaja/`.

### 4. Composição Landsat 2023 (Colab, com Earth Engine autenticado)
```python
!pip install -q earthengine-api geemap
import ee; ee.Authenticate(); ee.Initialize(project='SEU_PROJETO_GEE')

from src.data.gee_export_amaja import export_amaja_composite
export_amaja_composite(aoi_gpkg='data/raw/amaja/amaja_aoi.gpkg', year=2023)
```
Baixe o `.tif` exportado do Google Drive para `data/raw/amaja/landsat_amaja_2023.tif`.

### 5. Tiles pareados (imagem + máscara)
```bash
python src/data/tiles.py \
    --tif_path data/raw/amaja/landsat_amaja_2023.tif \
    --out_dir data/dataset_amaja/valid_images \
    --mask_gpkg data/raw/amaja/amaja_pivos.gpkg \
    --mask_out_dir data/dataset_amaja/valid_masks \
    --normalize percentile
```

### 6. Validação humana das máscaras
```bash
python src/data/review_tiles.py generate \
    --images_dir data/dataset_amaja/valid_images \
    --masks_dir data/dataset_amaja/valid_masks \
    --output_dir data/validation/amaja_review
```
Revise `data/validation/amaja_review/review_*.png` (composição + contorno da máscara
em amarelo) e preencha a coluna `decision` (`keep`/`discard`/`fix`) no
`review_manifest.csv` gerado. Em seguida:
```bash
python src/data/review_tiles.py apply \
    --images_dir data/dataset_amaja/valid_images \
    --masks_dir data/dataset_amaja/valid_masks \
    --manifest data/validation/amaja_review/review_manifest.csv \
    --output_images_dir data/dataset_amaja_final/valid_images \
    --output_masks_dir data/dataset_amaja_final/valid_masks
```

### 7. Avaliar a generalização geográfica
```bash
python src/main.py --mode validate --config config/config_amaja.yaml \
    --model runs/<run>/best_model.keras
```

---

## Configuração

Principais chaves de `config/config.yaml` (comentadas no próprio arquivo):

| Seção | Chave | Descrição |
|---|---|---|
| `data` | `min_fg_ratio` | Limiar de fração de SIPC para um tile contar como positivo |
| `data` | `desired_pos_ratio` | Proporção positivo/negativo alvo no balanceamento |
| `data` | `internal_val_fraction` | Fração do treino reservada para validação interna |
| `training` | `randommix` / `randommix_prob` | Ativa RandomMix e fração dos negativos mesclados |
| `training` | `loss.name` | `tversky` \| `dice` \| `bce` — as 3 funções comparadas no TCC |
| `training` | `loss.alpha` / `loss.beta` | Peso de FP/FN no Tversky loss (só se `loss.name: tversky`) |
| `gpu` | `memory_growth` / `memory_limit_mb` | Mutuamente exclusivos — `memory_growth` tem precedência |

`config/config_amaja.yaml` é bem mais enxuto: `--mode predict/export/validate` carregam
a arquitetura e os pesos direto do `.keras` salvo (nunca reconstroem a partir de um
`model:` no config), então só precisa de `data.balanced_path` (apontando ao dataset da
AMAJA gerado no Pipeline 2) e `training.loss.alpha`/`beta` (para reconstruir a loss
como custom_object ao carregar o modelo).

---

## Testes

```bash
pip install -r requirements-dev.txt
pytest tests/
```

Testes em módulos que não dependem de TensorFlow (`test_randommix.py`, `test_tiles.py`)
rodam em qualquer ambiente. Os que dependem (`test_dataset_balancer.py`,
`test_gpu_utils.py`, `test_unet.py`, `test_losses_metrics.py`,
`test_trainer_integration.py`) pulam automaticamente se TensorFlow não estiver
instalado.

Lint (`ruff`, config em `pyproject.toml`):
```bash
pip install ruff
ruff check src tests
```

O workflow em `.github/workflows/ci.yml` roda lint + testes (com TensorFlow instalado)
a cada push/PR na `main`.

---

## Solução de problemas

- **`ModuleNotFoundError: No module named 'tensorflow'` ao rodar um script de dados**:
  `amaja.py`, `tiles.py` e `review_tiles.py` não dependem de TensorFlow. Se o erro
  aparecer em `dataset_balancer.py`/`train.py`/`predict.py`, instale
  `requirements.txt`.
- **GPU não é detectada**: confira `nvidia-smi` e a instalação do CUDA compatível com
  `tensorflow==2.20.0` (ver `requirements.txt`).
- **`--mode predict/export/validate` reclamando de config ausente**: esses modos
  sempre esperam um `--config` válido (usado para reconstruir a loss/hiperparâmetros
  do modelo salvo).

---

## Referências

- Liu, X., He, W., Liu, W., Yin, G., & Zhang, H. (2023). Mapping annual center-pivot
  irrigated cropland in Brazil during the 1985-2021 period with cloud platforms and
  deep learning. *ISPRS Journal of Photogrammetry and Remote Sensing*, 205, 227-245.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for
  biomedical image segmentation. *MICCAI*.
- ANA (2021). Atlas da Irrigação: uso da água na agricultura irrigada, 2ª ed.

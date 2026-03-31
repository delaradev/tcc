# 🚜 SIPC Brazil - Identificação de Pivôs Centrais com Deep Learning

Pipeline para mapeamento automático de pivôs centrais no Brasil usando imagens Landsat e redes neurais U-Net.

Baseado no artigo: **Liu et al. (2023)** - *Mapping annual center-pivot irrigated cropland in Brazil during the 1985-2021 period with cloud platforms and deep learning* (ISPRS Journal of Photogrammetry and Remote Sensing)

https://doi.org/10.1016/j.isprsjprs.2023.10.007

https://zenodo.org/records/10046320

---


## 📋 Índice

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Fluxo de Trabalho](#fluxo-de-trabalho)
- [Configuração](#configuração)
- [Como Executar](#como-executar)
- [Monitoramento do Treinamento](#monitoramento-do-treinamento)
- [Resultados Esperados](#resultados-esperados)
- [Solução de Problemas](#solução-de-problemas)

---


## 📦 Pré-requisitos

### Hardware Recomendado
| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU | NVIDIA MX350 (4GB) | NVIDIA GTX 1060 (6GB) ou superior |
| RAM | 8GB | 16GB |
| Disco | 20GB | 50GB |

### Software
- Python 3.8 - 3.10
- CUDA 11.2+ (se usar GPU NVIDIA)
- TensorFlow 2.13.0
- Conta Google Earth Engine (para exportação de imagens)

---


## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/delaradev/irrigation-pivot-identifier.git
cd irrigation-pivot-identifier
```

### 2. Crie e ative o ambiente virtual
Criar ambiente
```bash
python -m venv venv
```

Ativar (Windows)
```bash
venv\Scripts\activate
```

Ativar (Linux/Mac)
```bash
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Verifique a instalação da GPU (opcional)
```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---


## 📁 Estrutura do Projeto
```bash
cpic-brazil/
│
├── config/
│   └── config.yaml # Configurações do treinamento
│
├── models/
│   ├── unet.py # Arquitetura U-Net
│   └── losses.py # Funções de loss (Tversky, Dice)
│
├── training/
│   ├── train.py # Pipeline de treinamento
│   ├── metrics.py # Métricas (IoU, Dice, Precision, Recall)
│   └── callbacks.py # Callbacks (PredictionSaver, GPU Monitor)
│
├── data/
│   └── dataset_builder.py # Construção do dataset (RandomMix)
│
├── utils/
│   └── gpu_utils.py # Configuração de GPU
│
├── runs/ # Resultados do treinamento (criado automaticamente)
│   └── CPIC_Brazil_YYYYMMDD_HHMMSS/
│       ├── best_model.keras # Melhor modelo (validação)
│       ├── history.json # Histórico de métricas
│       ├── config.yaml # Configuração usada
│       ├── tensorboard/ # Logs TensorBoard
│       └── predictions/ # Previsões por época
│
├── main.py # Ponto de entrada principal
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo

```

---


## 🔄 Fluxo de Trabalho

### ⚙️ Configuração
Edite ```config/config.yaml``` para ajustar parâmetros:

```bash
# Configurações principais
data:
  dataset_path: "data/dataset_balanced"  # Caminho do dataset
  tile_size: 512 # Tamanho dos tiles

model:
  img_size: 512 # Input size
  unet_base_filters: 16 # Filtros iniciais

training:
  batch_size: 1 # Para GPU com pouca VRAM
  epochs: 50 # Número de épocas
  learning_rate: 0.0001 # Learning rate
  randommix: true # Ativa RandomMix
  randommix_prob: 1.0 # Probabilidade de aplicar

gpu:
  memory_growth: true # Uso dinâmico de VRAM
  mixed_precision: true # Float16 (reduz memória ~50%)
```

---


## 🎯 Como Executar

### 1. Preparar os dados (Google Earth Engine)
Execute o script no GEE para exportar as composições anuais:

Baixe o arquivo exportado e coloque em ```data/raw/```

### 2. Gerar tiles (opcional, se não tiver dataset pronto)
```bash
python data/tile_generator.py \
    --input data/raw/landsat_cpic_2023.tif \
    --output data/tiles/ \
    --tile_size 512 \
    --overlap 32
```

### 3. Criar dataset balanceado
```bash
python data/create_balanced_dataset.py \
    --src_root data/tiles \
    --dst_root data/dataset_balanced \
    --min_fg_ratio 0.003 \
    --desired_pos_ratio 0.7
```

---


## 4. Treinar o modelo

### Treinamento completo
```bash
python main.py --config config/config.yaml --mode train
```

### Treinamento rápido (teste, apenas 5 épocas)
```bash
python main.py --config config/config.yaml --mode train --epochs 5
```
---


## 📊 Monitoramento do Treinamento

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Console Output

Durante o treinamento, você verá:
```bash
Epoch 1/50
GPU Info: {'name': 'NVIDIA GeForce MX350', 'memory_total': '4096 MiB', ...}
Dados: treino=1247, val=156
Iniciando treinamento...
1/1247 [>...] - ETA: 2:34:32 - loss: 0.7234 - iou_score: 0.1245
GPU Memory: 2345 MiB
...
```

---


### Métricas Monitoradas

| Métrica |	Descrição |	Alvo |
|---------|-----------|------|
IoU |	Intersection over Union |	> 0.90
Dice |	F1-score |	> 0.95
Precision |	TP / (TP + FP) |	> 0.95
Recall |	TP / (TP + FN) |	> 0.90

---


## 📈 Resultados Esperados
Com ```50 épocas```, ```batch_size=1```, ```img_size=512```:

| Métrica |	Valor Esperado |
|---------|----------------|
Val IoU |	0.92 - 0.95
Val Dice |	0.96 - 0.98
Val Precision |	0.97 - 0.99
Val Recall |	0.93 - 0.96
Tempo Total |	2.5 - 4 horas (MX350)

---


## 📚 Referências
Liu, X., He, W., Liu, W., Yin, G., & Zhang, H. (2023). Mapping annual center-pivot irrigated cropland in Brazil during the 1985-2021 period with cloud platforms and deep learning. ISPRS Journal of Photogrammetry and Remote Sensing, 205, 227-245.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention.

---


## 📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---


## 👥 Contribuições
Contribuições são bem-vindas! Abra uma issue ou pull request.

---


## 📧 Contato
Para dúvidas ou sugestões, abra uma issue no repositório.

---
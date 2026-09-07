"""CPIC Brazil — pacote raiz.

Deliberadamente sem re-exports aqui: importar `src` (ou qualquer subpacote) não deve
forçar o carregamento de dependências pesadas (TensorFlow) que só os módulos de
modelagem/treino precisam. Isso permite rodar scripts leves de dados/geoprocessamento
(src/data/amaja.py, tiles.py, ana_mask.py, review_tiles.py) sem TensorFlow instalado.
Importe sempre do submódulo específico, ex.: `from src.models.unet import build_unet`.
"""

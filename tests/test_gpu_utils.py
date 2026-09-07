"""Teste de regressão para o conflito corrigido em src/utils/gpu_utils.py: pedir
memory_growth e memory_limit_mb ao mesmo tempo (mutuamente exclusivos na mesma GPU no
TensorFlow) fazia a exceção ser engolida ANTES de ativar mixed_precision. Requer
TensorFlow — pula automaticamente se ausente."""
import pytest

tf = pytest.importorskip('tensorflow')

from src.utils.gpu_utils import configure_gpu


def test_no_gpu_returns_false(monkeypatch):
    monkeypatch.setattr(tf.config, 'list_physical_devices', lambda kind: [])
    assert configure_gpu(memory_growth=True, mixed_precision=True) is False


def test_conflicting_memory_options_still_enable_mixed_precision(monkeypatch):
    """Antes da correção, pedir memory_growth E memory_limit_mb ao mesmo tempo podia
    lançar RuntimeError dentro do loop de GPUs e pular a ativação de mixed_precision,
    que vinha depois no código. Agora mixed_precision é sempre aplicado, independente
    do resultado da configuração de memória por GPU."""
    fake_gpu = object()
    monkeypatch.setattr(tf.config, 'list_physical_devices', lambda kind: [fake_gpu])
    monkeypatch.setattr(tf.config.experimental, 'set_memory_growth', lambda gpu, enabled: None)

    policy_calls = []
    monkeypatch.setattr(tf.keras.mixed_precision, 'set_global_policy', policy_calls.append)

    configure_gpu(memory_growth=True, memory_limit_mb=40960, mixed_precision=True)

    assert policy_calls == ['mixed_float16']

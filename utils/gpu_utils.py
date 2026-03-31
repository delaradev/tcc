"""Utilitários para configuração de GPU"""
import tensorflow as tf
import os
from typing import Optional


def configure_gpu(
    memory_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    mixed_precision: bool = False
) -> bool:
    """

    Args:
        memory_growth: Ativa crescimento dinâmico de memória
        memory_limit_mb: Limite de memória em MB
        mixed_precision: Ativa mixed precision (float16)

    Returns:
        bool: True se GPU disponível, False caso contrário
    """
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("Nenhuma GPU encontrada. Usando CPU.")
        return False

    try:
        for gpu in gpus:
            print(f"GPU encontrada: {gpu}")

            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("Memória growth ativada")

            if memory_limit_mb:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )]
                )
                print(f"Limite de memória: {memory_limit_mb} MB")

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision ativada (float16)")

        return True

    except RuntimeError as e:
        print(f"Erro configurando GPU: {e}")
        return False


def get_gpu_info() -> dict:
    """Retorna informações sobre a GPU"""
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {
                'name': info[0],
                'memory_total': info[1],
                'memory_used': info[2] if len(info) > 2 else 'N/A'
            }
    except:
        pass

    return {'name': 'Unknown', 'memory_total': 'N/A', 'memory_used': 'N/A'}

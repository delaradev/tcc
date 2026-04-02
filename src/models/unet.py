"""Arquitetura U-Net conforme artigo Liu et al. 2023"""
import tensorflow as tf
from typing import Tuple


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Bloco convolucional: Conv2D + BatchNorm + ReLU (x2)"""
    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', use_bias=False, name=f'{name}_conv1'
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name}_relu1')(x)

    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', use_bias=False, name=f'{name}_conv2'
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = tf.keras.layers.ReLU(name=f'{name}_relu2')(x)

    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    base_filters: int = 16
) -> tf.keras.Model:
    """
    Constrói modelo U-Net conforme o artigo

    Args:
        input_shape: Formato da entrada (height, width, channels)
        base_filters: Número de filtros na primeira camada

    Returns:
        Modelo Keras compilado
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')

    # Encoder (downsampling)
    c1 = conv_block(inputs, base_filters, 'enc1')
    p1 = tf.keras.layers.MaxPooling2D(name='pool1')(c1)

    c2 = conv_block(p1, base_filters * 2, 'enc2')
    p2 = tf.keras.layers.MaxPooling2D(name='pool2')(c2)

    c3 = conv_block(p2, base_filters * 4, 'enc3')
    p3 = tf.keras.layers.MaxPooling2D(name='pool3')(c3)

    c4 = conv_block(p3, base_filters * 8, 'enc4')
    p4 = tf.keras.layers.MaxPooling2D(name='pool4')(c4)

    # Bottleneck
    bottleneck = conv_block(p4, base_filters * 16, 'bottleneck')

    # Decoder (upsampling)
    u4 = tf.keras.layers.UpSampling2D(name='up4')(bottleneck)
    u4 = tf.keras.layers.Concatenate(name='cat4')([u4, c4])
    d4 = conv_block(u4, base_filters * 8, 'dec4')

    u3 = tf.keras.layers.UpSampling2D(name='up3')(d4)
    u3 = tf.keras.layers.Concatenate(name='cat3')([u3, c3])
    d3 = conv_block(u3, base_filters * 4, 'dec3')

    u2 = tf.keras.layers.UpSampling2D(name='up2')(d3)
    u2 = tf.keras.layers.Concatenate(name='cat2')([u2, c2])
    d2 = conv_block(u2, base_filters * 2, 'dec2')

    u1 = tf.keras.layers.UpSampling2D(name='up1')(d2)
    u1 = tf.keras.layers.Concatenate(name='cat1')([u1, c1])
    d1 = conv_block(u1, base_filters, 'dec1')

    # Output
    outputs = tf.keras.layers.Conv2D(
        1, 1, activation='sigmoid', name='output'
    )(d1)

    model = tf.keras.Model(inputs, outputs, name='unet')
    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Retorna resumo do modelo como string"""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)

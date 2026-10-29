"""
UNet architecture for crack segmentation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import UNET_IMG_SIZE, CHANNELS


def conv_block(x, filters: int, name_prefix: str):
    """Two Conv2D → BN → ReLU layers."""
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      name=f'{name_prefix}_c1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_act1')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      name=f'{name_prefix}_c2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_act2')(x)
    return x


def encoder_block(x, filters: int, name_prefix: str):
    """Conv block + MaxPool. Returns (skip_connection, pooled)."""
    skip = conv_block(x, filters, name_prefix)
    pooled = layers.MaxPooling2D(2, name=f'{name_prefix}_pool')(skip)
    pooled = layers.Dropout(0.2, name=f'{name_prefix}_drop')(pooled)
    return skip, pooled


def decoder_block(x, skip, filters: int, name_prefix: str):
    """Upsample → concat skip → conv block."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same',
                               name=f'{name_prefix}_up')(x)
    x = layers.Concatenate(name=f'{name_prefix}_cat')([x, skip])
    x = layers.Dropout(0.2, name=f'{name_prefix}_drop')(x)
    x = conv_block(x, filters, name_prefix)
    return x


def build_unet_segmentor(input_shape=(UNET_IMG_SIZE[0], UNET_IMG_SIZE[1], CHANNELS),
                          filters=(64, 128, 256, 512)) -> keras.Model:
    """
    Standard U-Net for binary crack segmentation.
    Output: single-channel sigmoid mask (0=background, 1=crack).
    """
    inputs = keras.Input(shape=input_shape, name="seg_input")

    # ── Encoder (Contracting Path) ───────────────────────────────────────────
    s1, p1 = encoder_block(inputs, filters[0], 'enc1')
    s2, p2 = encoder_block(p1,     filters[1], 'enc2')
    s3, p3 = encoder_block(p2,     filters[2], 'enc3')
    s4, p4 = encoder_block(p3,     filters[3], 'enc4')

    # ── Bottleneck ───────────────────────────────────────────────────────────
    b = conv_block(p4, filters[3] * 2, 'bottleneck')

    # ── Decoder (Expanding Path) ─────────────────────────────────────────────
    d4 = decoder_block(b,  s4, filters[3], 'dec4')
    d3 = decoder_block(d4, s3, filters[2], 'dec3')
    d2 = decoder_block(d3, s2, filters[1], 'dec2')
    d1 = decoder_block(d2, s1, filters[0], 'dec1')

    # ── Output ───────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation='sigmoid', name='seg_output')(d1)

    model = keras.Model(inputs=inputs, outputs=outputs, name='CrackUNet')
    return model


if __name__ == "__main__":
    model = build_unet_segmentor()
    model.summary()

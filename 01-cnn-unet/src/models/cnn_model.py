"""
CNN architecture for crack binary classification + intensity grading.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import CNN_IMG_SIZE, CHANNELS


def build_cnn_classifier(input_shape=(CNN_IMG_SIZE[0], CNN_IMG_SIZE[1], CHANNELS),
                          num_classes: int = 1) -> keras.Model:
    """
    Custom CNN for crack classification.
    Returns probability of crack presence (sigmoid output).

    Architecture:
      - 4 convolutional blocks (Conv → BN → ReLU → MaxPool)
      - Global Average Pooling
      - Dense head with Dropout
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # ── Block 1 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv1a')(inputs)
    x = layers.BatchNormalization(name='bn1a')(x)
    x = layers.Activation('relu', name='act1a')(x)
    x = layers.Conv2D(32, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv1b')(x)
    x = layers.BatchNormalization(name='bn1b')(x)
    x = layers.Activation('relu', name='act1b')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)

    # ── Block 2 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv2a')(x)
    x = layers.BatchNormalization(name='bn2a')(x)
    x = layers.Activation('relu', name='act2a')(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv2b')(x)
    x = layers.BatchNormalization(name='bn2b')(x)
    x = layers.Activation('relu', name='act2b')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.Dropout(0.25, name='drop2')(x)

    # ── Block 3 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv3a')(x)
    x = layers.BatchNormalization(name='bn3a')(x)
    x = layers.Activation('relu', name='act3a')(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv3b')(x)
    x = layers.BatchNormalization(name='bn3b')(x)
    x = layers.Activation('relu', name='act3b')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)

    # ── Block 4 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv4a')(x)
    x = layers.BatchNormalization(name='bn4a')(x)
    x = layers.Activation('relu', name='act4a')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4), name='conv4b')(x)
    x = layers.BatchNormalization(name='bn4b')(x)
    x = layers.Activation('relu', name='act4b')(x)
    x = layers.MaxPooling2D(2, name='pool4')(x)
    x = layers.Dropout(0.3, name='drop4')(x)

    # ── Classification head ──────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc')(x)
    x = layers.Dropout(0.5, name='drop_fc')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='CrackCNN')
    return model


if __name__ == "__main__":
    model = build_cnn_classifier()
    model.summary()

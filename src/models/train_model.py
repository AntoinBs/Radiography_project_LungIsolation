import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import time
from pathlib import Path

import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.model_selection import train_test_split

from src.features.utils import load_images

def down_block(x, filters):
    """
    Downsampling block for the U-Net model.
    """
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def up_block(x, skip, filters):
    """
    Upsampling block for the U-Net model.
    """
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip])
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def build_unet(input_shape, first_layer_filters):
    """
    Build the U-Net model.
    """
    inputs = Input(shape=input_shape)

    # Downsampling path
    temp1, x = down_block(inputs, first_layer_filters)
    temp2, x = down_block(x, first_layer_filters * 2)
    temp3, x = down_block(x, first_layer_filters * 4)
    temp4, x = down_block(x, first_layer_filters * 8)

    # Bottleneck
    bottleneck = Conv2D(first_layer_filters * 16, (3, 3), activation='relu', padding='same')(x)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Conv2D(first_layer_filters * 16, (3, 3), activation='relu', padding='same')(bottleneck)
    bottleneck = BatchNormalization()(bottleneck)

    # Upsampling path
    x = up_block(bottleneck, temp4, first_layer_filters * 8)
    x = up_block(x, temp3, first_layer_filters * 4)
    x = up_block(x, temp2, first_layer_filters * 2)
    x = up_block(x, temp1, first_layer_filters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    return model

@keras.saving.register_keras_serializable()
class IoUMetric(keras.metrics.Metric):
    def __init__(self, name='iou_metric', threshold=0.5, **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        iou = intersection / (union + tf.keras.backend.epsilon())

        self.total_iou.assign_add(iou)
        self.count.assign_add(1)

    def result(self):
        return self.total_iou / (self.count + tf.keras.backend.epsilon())

    def reset_states(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(IoUMetric, self).get_config()
        config.update({"threshold": self.threshold})
        return config


if __name__ == "__main__":
    # Chargement des métadonnées
    metadata = pd.read_csv(r'./data/processed/metadata.csv')

    # Extraction des chemins d'images et de masques
    image_paths = metadata["IMAGE_URL"].values
    mask_paths = metadata["MASK_URL"].values

    # Séparation des données en ensembles d'entraînement, de validation et de test
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(image_paths, mask_paths, test_size=0.3, random_state=42)
    _, X_val, _, y_val = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=42)

    # Transformation des chemins d'images et de masques en dataset Tensorflow, puis application de la fonction de chargement
    # de données sur chaque élément du dataset

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(lambda x, y: load_images(x, y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(lambda x, y: load_images(x, y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    # Mélange des données dans les datasets,

    BATCH_SIZE = 32

    train_ds = (
        train_ds
        .shuffle(buffer_size=100)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Définition des callbacks

    model_path = r"./models/lung_seg.keras"

    modelcheckpoint = ModelCheckpoint(filepath=model_path,
                                    monitor='val_iou_metric',
                                    verbose=1,
                                    save_best_only=True,
                                    mode="max")

    earlystop = EarlyStopping(monitor="val_loss",
                            mode="min",
                            min_delta=0,
                            patience=8,
                            verbose=1,
                            restore_best_weights=True)

    reducelr = ReduceLROnPlateau(monitor="val_loss",
                                min_lr=0.00001,
                                patience=4,
                                factor=0.5,
                                cooldown=2,
                                verbose=1)

    # Construction du modèle U-Net
    input_shape = (256, 256, 1)
    unet = build_unet(input_shape, 16)

    # Compilation du modèle
    unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', IoUMetric(name='iou_metric')])

    if tf.config.list_physical_devices('GPU'):
        print("GPU is available. Training on GPU.")
        with tf.device('/GPU:0'):
        # Entraînement du modèle
            history = unet.fit(train_ds,
                            validation_data=val_ds,
                            epochs=50,
                            callbacks=[modelcheckpoint, earlystop, reducelr],
                            verbose=1)
    else:
        print("GPU is not available. Training on CPU.")
        history = unet.fit(train_ds,
                            validation_data=val_ds,
                            epochs=50,
                            callbacks=[modelcheckpoint, earlystop, reducelr],
                            verbose=1)
    
    # Sauvegarde du modèle
    unet.save(model_path)
    print(f"Model saved to {model_path}")
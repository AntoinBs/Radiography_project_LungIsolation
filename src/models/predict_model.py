
from src.models.train_model import IoUMetric

from matplotlib import pyplot as plt
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model

def load_images(image_path, mask_path):
    """
    Load images and masks from the given paths.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((256, 256, 1))

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask.set_shape((256, 256, 1))

    return image, mask

model = load_model('models/lung_seg.keras', custom_objects={"IoUMetric": IoUMetric})

image, mask = load_images('data/processed/images/image-2.png', 'data/processed/masks/mask-2.png')

print(f"Image shape: {image.shape}, type: {image.dtype}, object type: {type(image)}")
prediction = model.predict(tf.expand_dims(image, axis=0))
prediction = tf.squeeze(prediction, axis=0)  # Remove batch dimension
prediction = tf.where(prediction > 0.5, 1.0, 0.0)  # Binarize the prediction

plt.imshow(prediction, cmap='gray')
plt.axis('off')
plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
plt.close()
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model

from matplotlib import pyplot as plt
import cv2

def load_images(image_path, mask_path):
    """
    Load images and masks from the given paths.
    """
    image = tf.io.read_file(image_path) # Read the image file
    image = tf.image.decode_png(image, channels=1) # Decode the image to grayscale
    image = tf.cast(image, tf.float32) / 255.0 # Normalize the image to [0, 1]
    image.set_shape((256, 256, 1)) # Ensure the shape of the image

    mask = tf.io.read_file(mask_path) # Read the mask file
    mask = tf.image.decode_png(mask, channels=1) # Decode the mask to grayscale
    mask = tf.cast(mask, tf.float32) / 255.0 # Normalize the mask to [0, 1]
    mask.set_shape((256, 256, 1)) # Ensure the shape of the mask

    return image, mask

def prepare_image_seg(file):
    """
    Prepare an image for segmentation prediction.
    """
    image = tf.image.decode_image(file, channels=1) # Decode the image to grayscale
    image = tf.cast(image, tf.float32) / 255.0 # Normalize the image to [0, 1]
    image = tf.image.resize(image, (256, 256)) # Resize the image to the target size
    image.set_shape((256, 256, 1)) # Ensure the shape of the image
    return tf.expand_dims(image, axis=0)  # Add batch dimension

def prepare_image_class(image, mask):
    """
    Prepare an image and its mask for classification prediction.
    This function applies the mask to the image, resizes both to 299x299,
    and converts them to RGB format for visualization.
    """
    mask = np.where(mask > 0.5, 1, 0)  # Binariser le masque
    mask = tf.squeeze(mask, axis=0)  # Enlever la dimension de canal
    mask = tf.image.resize(mask, (299, 299))
    mask = mask.numpy()  # Convertir en numpy array

    image = tf.squeeze(image, axis=0)  # Enlever la dimension de canal
    image = tf.image.resize(image, (299, 299))
    image = tf.cast(image, tf.float32) * 255.0  # Normaliser l'image
    image = image.numpy()  # Convertir en numpy array

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.uint8)  # Convertir en RGB et en uint8
    mask_rgb = np.concatenate([mask*255.0, np.zeros(mask.shape), np.zeros(mask.shape)], axis=-1).astype(np.uint8)  # Convertir le masque en RGB et en uint8

    image_mask_rgb = cv2.addWeighted(image_rgb, 0.7, mask_rgb, 0.3, 0)  # Fusionner l'image et le masque

    image = np.multiply(image, mask)  # Appliquer le masque
    image = image.astype(np.uint8)  # Convertir en float32
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convertir en RGB
    
    return np.expand_dims(image, axis=0), image_mask_rgb  # Ajouter batch dimension

def grad_cam(img, raw_image, model, base_model, layer_name : str):
    # Sélection de la couche dans le bon modèle
    layer = base_model.get_layer(layer_name)
    grad_model = Model(inputs=base_model.input, outputs=[layer.output, base_model.output])


    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Calcul des gradients par rapport aux activations de la couche convolutionnelle
    grads = tape.gradient(loss, conv_outputs)

    # Moyenne pondérée des gradients pour chaque canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des activations par les gradients calculés
    conv_outputs = conv_outputs[0] # Supprimer la dimension batch
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalisation de la heatmap
    heatmap = tf.maximum(heatmap, 0) # Se concentrer uniquement sur les valeurs positives
    heatmap /= tf.math.reduce_max(heatmap) # Normalisation entre 0 et 1
    heatmap = heatmap.numpy() # Convertir en tableau numpy pour la visualisation

    # Redimensionner la heatmap pour correspondre à la taille de l'image d'origine
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (raw_image.shape[1], raw_image.shape[2])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1) # Supprimer la dimension de taille 1 à la fin du tableau

    # Colorier la heatmap avec une palette (par exemple "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] # Récupérer les canaux R, G, B
    
    raw_image = tf.image.resize(raw_image, heatmap_colored.shape[:-1])  # Redimensionner l'image brute pour correspondre à la taille de la heatmap
    raw_image = raw_image.numpy()  # Convertir en tableau numpy et en uint8

    # Superposer la heatmap à l'image
    superposed_img = heatmap_colored*0.4 + raw_image*0.6 # Superposer la heatmap à l'image brute avec une opacité de 0.4
    image_grad_cam = np.clip(superposed_img, 0, 1) # Garantit que toutes les valeurs de l'image finale se situent entre
    # 0 et 1

    return image_grad_cam
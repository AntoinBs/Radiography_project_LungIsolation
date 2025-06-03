import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from io import BytesIO
import uuid
from typing import Dict
from PIL import Image

import cv2
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, SeparableConv2D

from src.features.utils import prepare_image_seg, prepare_image_class, grad_cam
from src.models.train_model import IoUMetric

# Load the trained models
segmentation_model = load_model('models/lung_seg.keras', custom_objects={"IoUMetric": IoUMetric})
classification_model = load_model('models/lung_class.keras')

# Initialize stores for images and predictions from sessions_ids
image_for_seg_store: Dict[str, np.ndarray] = {}
image_for_class_store: Dict[str, np.ndarray] = {}
predictions_store: Dict[str, np.ndarray] = {}
image_masked_rgb_store: Dict[str, np.ndarray] = {}

class_names = ['COVID', 'LUNG_OPACITY', 'NORMAL', 'VIRAL_PNEUMONIA'] # Class names for classification

app = FastAPI() # Create FastAPI app instance

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplace par ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lung Pathology Classification API"}


@app.post("/upload/image/")
async def upload_and_preprocess_image(file: UploadFile = File(...)):
    """
    Upload and preprocess an image for segmentation.
    Returns a unique session ID for next operations.
    """
    try:
        session_id = str(uuid.uuid4()) # Generate a unique session ID

        contents = await file.read()
        processed_image = prepare_image_seg(contents)

        # Store the processed image in memory
        image_for_seg_store[session_id] = processed_image
    
        return {
            "session_id": session_id,
            "status": "Ready for segmentation"
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/segmentation/{session_id}")
async def predict_segmentation(session_id: str):
    """
    Predict segmentation mask for the uploaded image using the session ID.
    """
    try:
        if session_id not in image_for_seg_store:
            return {"error": "Session ID not found. Please upload an image first."}

        processed_image = image_for_seg_store[session_id] # load the processed image from store

        prediction = segmentation_model.predict(processed_image) # Predict the segmentation mask

        prediction, image_masked_rgb = prepare_image_class(processed_image, prediction) # Prepare the image and mask for classification
        
        image_for_class_store[session_id] = prediction # Store the prediction
        image_masked_rgb_store[session_id] = image_masked_rgb # Store the masked RGB image

        return {
            "session_id": session_id,
            "status": "Segmentation prediction completed, ready for classification"
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.post("/predict/classification/{session_id}")
async def predict_classification(session_id: str):
    """
    Predict the pathology for the segmented image.
    """
    try:
        if session_id not in image_for_seg_store:
            return {"error": "Session ID not found. Please upload an image first."}

        processed_image = image_for_class_store[session_id] # Load the processed image for classification

        prediction = classification_model.predict(processed_image) # Predict the classification
        prediction = np.argmax(prediction, axis=1)  # Get the class with the highest probability
        prediction = class_names[prediction[0]]  # Map to class name
        
        predictions_store[session_id] = prediction # Store the prediction

        return {
            "session_id": session_id,
            "status": "Classification prediction completed"
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/result/prediction/{session_id}")
async def get_predictions(session_id: str):
    """
    Get the predictions for a given session ID.
    """
    try:
        if session_id not in predictions_store:
            return {"error": "Session ID not found. Please perform classification first."}

        prediction = predictions_store[session_id]
        return {
            "session_id": session_id,
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/result/segmentation/{session_id}")
async def get_segmentation_result(session_id: str):
    """
    Get the segmentation result for a given session ID.
    """
    try:
        if session_id not in image_masked_rgb_store:
            return {"error": "Session ID not found. Please perform segmentation first."}

        prediction = image_masked_rgb_store[session_id] # Load the masked RGB image
        
        prediction = cv2.resize(prediction, (256, 256))

        # Convert the prediction to a PIL image for response
        pil_image = Image.fromarray(prediction, mode="RGB")
        img_buffer = BytesIO() # Create a BytesIO buffer to hold the image data
        pil_image.save(img_buffer, format="PNG") # Save the image to the buffer
        img_buffer.seek(0)

        # Return the image as a streaming response
        return StreamingResponse(img_buffer, media_type="image/png", headers={"Content-Disposition": f"attachment; filename=segmentation_result_{session_id}.png"})

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/result/classification/{session_id}")
async def get_classification_result(session_id: str):
    """
    Get the classification gradcam for a given session ID.
    """
    try:
        if session_id not in predictions_store:
            return {"error": "Session ID not found. Please perform classification first."}
        
        
        image = image_for_seg_store[session_id] # Load the original image

        if predictions_store[session_id] != "NORMAL":
            "Get gradcam if classification is not NORMAL"

            processed_image = image_for_class_store[session_id] # Load the processed image for classification

            base_model = classification_model.get_layer("xception")  # Get the base model

            conv_layers = [layer.name for layer in base_model.layers if (isinstance(layer, Conv2D) or isinstance(layer, SeparableConv2D))][-1]

            gradcam = grad_cam(processed_image, image, classification_model, base_model, conv_layers) # Get the gradcam
        else:
            gradcam = cv2.cvtColor(image[0, :, :, 0].numpy(), cv2.COLOR_GRAY2RGB)  # Convert to RGB if NORMAL
            gradcam = np.expand_dims(gradcam, axis=0)  # Add batch dimension

        gradcam = np.squeeze(gradcam, axis=0) * 255 # Remove batch dimension and scale to [0, 255]
        gradcam = gradcam.astype(np.uint8)  # Convert to uint8

        # Convert the prediction to a PIL image for response
        pil_image = Image.fromarray(gradcam, mode="RGB")
        img_buffer = BytesIO() # Create a BytesIO buffer to hold the image data
        pil_image.save(img_buffer, format="PNG") # Save the image to the buffer
        img_buffer.seek(0)

        # Return the image as a streaming response
        return StreamingResponse(img_buffer, media_type="image/png", headers={"Content-Disposition": f"attachment; filename=segmentation_result_{session_id}.png"})

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/image/{session_id}")
async def get_raw_image(session_id: str):
    """
    Get the raw image for a given session ID.
    """
    try:
        if session_id not in image_for_seg_store:
            return {"error": "Session ID not found. Please perform segmentation first."}

        image = image_for_seg_store[session_id]
        image = tf.squeeze(image, axis=[0, -1])  # Remove batch dimension and channel dimension
        image = image.numpy() * 255.0  # Convert to numpy array and scale to [0, 255]
        image = image.astype(np.uint8)  # Convert to uint8

        pil_image = Image.fromarray(image, mode="L") # Convert to PIL image in grayscale mode
        img_buffer = BytesIO() # Create a BytesIO buffer to hold the image data
        pil_image.save(img_buffer, format="PNG") # Save the image to the buffer
        img_buffer.seek(0)

        # Return the image as a streaming response
        return StreamingResponse(img_buffer, media_type="image/png", headers={"Content-Disposition": f"attachment; filename=raw_image_{session_id}.png"})

    except Exception as e:
        return {"error": str(e)}
    
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its associated data.
    """
    try:
        if session_id in image_for_seg_store:
            del image_for_seg_store[session_id]
        if session_id in image_for_class_store:
            del image_for_class_store[session_id]
        if session_id in predictions_store:
            del predictions_store[session_id]
        if session_id in image_masked_rgb_store:
            del image_masked_rgb_store[session_id]

        return {"status": "Session deleted successfully"}

    except Exception as e:
        return {"error": str(e)}
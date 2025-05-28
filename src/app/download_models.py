import os
import boto3
import logging
from botocore.exceptions import ClientError
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models_from_s3(bucket_name, models_info, dest_dir='./models/'):
    """
    Download models from an S3 bucket.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        models_info (list): List of dictionaries containing model information
                           [{'s3_key': 'path/in/s3/model.keras', 'local_filename': 'model.keras'}, ...]
        dest_dir (str): Local directory to save models
    
    Returns:
        bool: True if all downloads were successful, False otherwise
    """
    # Create S3 client
    try:
        s3_client = boto3.client('s3')
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return False
    
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Track overall success
    all_successful = True
    
    # Download each model
    for model in models_info:
        s3_key = model['s3_key']
        local_path = os.path.join(dest_dir, model['local_filename'])
        
        logger.info(f"Downloading {s3_key} to {local_path}...")
        
        try:
            s3_client.download_file(bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded {local_path}")
        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            all_successful = False
        except Exception as e:
            logger.error(f"Unexpected error downloading {s3_key}: {e}")
            all_successful = False
    
    return all_successful

if __name__ == "__main__":
    # S3 bucket name
    BUCKET_NAME = "lung-pathology-radiography-models"
    
    # Define models to download
    MODELS = [
        {
            "s3_key": "lung_seg.keras",
            "local_filename": "lung_seg.keras"
        },
        {
            "s3_key": "lung_class.keras", 
            "local_filename": "lung_class.keras"
        }
    ]
    
    # Destination directory
    DEST_DIR = "./models/"
    
    # Download models
    success = download_models_from_s3(BUCKET_NAME, MODELS, DEST_DIR)
    
    if not success:
        logger.error("Failed to download one or more models")
        sys.exit(1)
    else:
        logger.info("All models downloaded successfully")
        sys.exit(0)
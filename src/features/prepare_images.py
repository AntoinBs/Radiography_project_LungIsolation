import cv2

import pandas as pd
import os

def prepare_image(image_path:str, output_path:str, resize:tuple=None) -> None:
    """
    Prepares images and masks for training by resizing them.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        resize (tuple, optional): New size for the image.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image if resize argument is provided
    if resize:
        image = cv2.resize(image, resize)

    # Save the processed image
    cv2.imwrite(output_path, image)  # Convert back to uint8 for saving


if __name__ == "__main__":
    # Read the metadata file
    metadata = pd.read_csv(r".\data\raw\metadata.csv")
    metadata['IMAGE_URL'] = ''
    metadata['MASK_URL'] = ''
    parent_dir = r".\data\raw"

    for index, (image_name, pathology) in enumerate(zip(metadata['FILE NAME'], metadata['PATHOLOGY'])):
        for folder in ["images", "masks"]:
            image_folder = image_name.split("-")[0] # get the folder name
            raw_path = os.path.join(parent_dir, image_folder, folder, image_name+".png") # construct the path
            output_folder = os.path.join(r".\data\processed", pathology, folder) # make the output folder
            new_file_name = pathology + '-' + image_name.split("-")[1] + ".png"
            processed_path = os.path.join(output_folder, new_file_name) # construct the output path
            if not os.path.exists(output_folder):
                # create the output folder if it doesn't exist
                os.makedirs(output_folder)

            if folder == "images":
                prepare_image(raw_path, processed_path)
                metadata.loc[index, "IMAGE_URL"] = processed_path # add the image URL in the metadata
            else:
                prepare_image(raw_path, processed_path, resize=(299, 299))
                metadata.loc[index, "MASK_URL"] = processed_path # add the mask URL in the metadata

            metadata.loc[index, "FILE NAME"] = new_file_name.split(".")[0] # update the file name in the metadata
    
    metadata_output = r".\data\processed\metadata.csv"
    metadata.to_csv(metadata_output, index=False) # save the metadata
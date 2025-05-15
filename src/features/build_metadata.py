import pandas as pd

import os

def build_metadata(path: str) -> pd.DataFrame:
    """
    Build metadata for the dataset from the 4 raw datasets.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: Metadata DataFrame.
    """
    # Get all files in the directory
    files = os.listdir(path)

    # Create a list to hold metadata
    df = pd.DataFrame()

    # Iterate through each file and extract metadata
    for file in files:
        if file.endswith('.xlsx'):
            #pathology = file.split('.')[0].upper().replace(' ', '_') # get pathology name

            # Read the metadata file
            file_path = os.path.join(path, file)
            metadata = pd.read_excel(file_path)

            if file.split('.')[0] == "Normal":
                # Add a column for the URL
                metadata['FILE NAME'] = metadata['FILE NAME'].apply(lambda x: 'Normal' + '-' + x.split('-')[1])

            # Extract metadata (e.g., filename, duration, etc.)
            df = pd.concat([df, metadata], ignore_index=True, axis=0)
    
    df.drop(columns=['URL'], inplace=True) # drop the URL column

    return df

if __name__ == "__main__":
    metadata_input = "./data/raw"
    metadata_df = build_metadata(metadata_input)
    metadata_output = "./data/raw/metadata.csv"

    # Check if the output directory exists, if not create it
    if not os.path.exists(os.path.dirname(metadata_output)):
        os.makedirs(os.path.dirname(metadata_output))
    
    metadata_df.to_csv(metadata_output, index=False)
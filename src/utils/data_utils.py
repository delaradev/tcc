import zipfile
import os
from pathlib import Path

def extract_dataset():
    """Extract zip file from data/raw/dataset directory"""
    data_raw_path = Path("data/raw/dataset")
    
    # Find zip file
    zip_files = list(data_raw_path.glob("*.zip"))
    
    if not zip_files:
        print("No zip file found in data/raw/dataset")
        return
    
    zip_file_path = zip_files[0]
    extract_path = data_raw_path
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Successfully extracted {zip_file_path.name}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_file_path.name} is not a valid zip file")
    except Exception as e:
        print(f"Error extracting file: {e}")

if __name__ == "__main__":
    extract_dataset()
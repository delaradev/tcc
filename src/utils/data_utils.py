#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DATA UTILITIES - EXTRACTION AND ORGANIZATION
================================================================================

Utilities for extracting and organizing CPIC dataset from Zenodo.

Data structure:
    data/
        raw/dataset/zenodo_cpic_brazil.zip  -> original download
        dataset/                             -> extracted (train/valid)

Author: Emanuel de Lara Ruas
Version: 1.0.0
================================================================================
"""

import zipfile
import shutil
from pathlib import Path
from typing import Optional


def extract_zenodo_dataset(
    zip_path: Optional[Path] = None,
    extract_to: Optional[Path] = None,
    remove_zip: bool = False
) -> Path:
    """
    Extract Zenodo CPIC dataset from raw folder to dataset folder

    Args:
        zip_path: Path to zip file (default: data/raw/dataset/*.zip)
        extract_to: Destination directory (default: data/dataset/)
        remove_zip: Remove zip file after extraction

    Returns:
        Path to extracted dataset directory
    """
    # Default paths
    if zip_path is None:
        raw_dataset_dir = Path("data/raw/dataset")
        zip_files = list(raw_dataset_dir.glob("*.zip"))

        if not zip_files:
            raise FileNotFoundError(
                "No zip file found in data/raw/dataset/\n"
                "Please download dataset from Zenodo and place it there."
            )

        zip_path = zip_files[0]
        print(f"Found zip file: {zip_path.name}")

    if extract_to is None:
        extract_to = Path("data/dataset")

    # Ensure destination exists
    extract_to.mkdir(parents=True, exist_ok=True)

    # Extract
    print(f"Extracting to: {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Successfully extracted {zip_path.name}")

        # Optional: remove zip
        if remove_zip:
            zip_path.unlink()
            print(f"🗑️ Removed zip file: {zip_path}")

    except zipfile.BadZipFile:
        raise RuntimeError(f"File is not a valid zip: {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")

    return extract_to


def verify_dataset_structure(dataset_path: Path = Path("data/dataset")) -> bool:
    """
    Verify dataset has expected structure

    Expected:
        dataset_path/
            train_images/
            train_masks/
            valid_images/
            valid_masks/
    """
    required_dirs = ['train_images', 'train_masks',
                     'valid_images', 'valid_masks']

    print("\n📁 Verifying dataset structure:")
    all_exist = True

    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        exists = dir_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_name}/")

        if not exists:
            all_exist = False

    if all_exist:
        print("\n✅ Dataset structure is correct!")

        # Count samples
        train_images = len(list((dataset_path / 'train_images').glob('*.png')))
        train_masks = len(list((dataset_path / 'train_masks').glob('*.png')))
        valid_images = len(list((dataset_path / 'valid_images').glob('*.png')))
        valid_masks = len(list((dataset_path / 'valid_masks').glob('*.png')))

        print(f"\n📊 Sample counts:")
        print(f"  Train images: {train_images}")
        print(f"  Train masks:  {train_masks}")
        print(f"  Valid images: {valid_images}")
        print(f"  Valid masks:  {valid_masks}")

    else:
        print("\n⚠️ Dataset structure is incomplete!")
        print("   Run extract_zenodo_dataset() first.")

    return all_exist


def get_dataset_info() -> dict:
    """Get information about dataset structure"""
    dataset_path = Path("data/dataset")
    balanced_path = Path("data/dataset_balanced")

    info = {
        'raw_exists': Path("data/raw/dataset").exists(),
        'dataset_exists': dataset_path.exists(),
        'balanced_exists': balanced_path.exists(),
    }

    if dataset_path.exists():
        info['dataset'] = {
            'train_images': len(list(dataset_path.glob('train_images/*.png'))),
            'train_masks': len(list(dataset_path.glob('train_masks/*.png'))),
            'valid_images': len(list(dataset_path.glob('valid_images/*.png'))),
            'valid_masks': len(list(dataset_path.glob('valid_masks/*.png'))),
        }

    if balanced_path.exists():
        info['balanced'] = {
            'train_images': len(list(balanced_path.glob('train_images/*.png'))),
            'train_masks': len(list(balanced_path.glob('train_masks/*.png'))),
            'valid_images': len(list(balanced_path.glob('valid_images/*.png'))),
            'valid_masks': len(list(balanced_path.glob('valid_masks/*.png'))),
        }

    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPIC Dataset utilities")
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract dataset from raw folder'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset structure'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset information'
    )
    parser.add_argument(
        '--remove_zip',
        action='store_true',
        help='Remove zip after extraction'
    )

    args = parser.parse_args()

    if args.extract:
        extract_zenodo_dataset(remove_zip=args.remove_zip)

    if args.verify:
        verify_dataset_structure()

    if args.info:
        info = get_dataset_info()
        print("\n📊 Dataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    if not any([args.extract, args.verify, args.info]):
        print("Usage:")
        print("  python data_utils.py --extract    # Extract dataset")
        print("  python data_utils.py --verify     # Verify structure")
        print("  python data_utils.py --info       # Show info")

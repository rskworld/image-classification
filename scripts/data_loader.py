"""
===================================================================================
    Project: Image Classification Dataset
    Description: Data loading utilities for the image classification dataset.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
import cv2


class ImageDataLoader:
    """
    A comprehensive data loader for the image classification dataset.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        grayscale: bool = False
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the dataset directory
            image_size: Target size for images (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert images to grayscale
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize = normalize
        self.grayscale = grayscale
        self.categories = self._get_categories()
        self.num_classes = len(self.categories)
        
    def _get_categories(self) -> List[str]:
        """Get list of category names from directory structure."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        categories = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        return categories
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array or None if loading fails
        """
        try:
            # Load image using PIL
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB' and not self.grayscale:
                img = img.convert('RGB')
            elif self.grayscale:
                img = img.convert('L')
            
            # Resize
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize if required
            if self.normalize:
                img_array = img_array.astype(np.float32) / 255.0
                
            return img_array
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_dataset(
        self,
        subset: str = 'train',
        max_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load the entire dataset or a subset.
        
        Args:
            subset: Which subset to load ('train', 'validation', 'test')
            max_samples_per_class: Maximum samples per class (for debugging)
            
        Returns:
            Tuple of (images, labels, category_names)
        """
        subset_dir = os.path.join(self.data_dir, subset)
        
        if not os.path.exists(subset_dir):
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        images = []
        labels = []
        
        print(f"Loading {subset} dataset from {subset_dir}...")
        
        for idx, category in enumerate(tqdm(self.categories, desc="Categories")):
            category_path = os.path.join(subset_dir, category)
            
            if not os.path.exists(category_path):
                continue
                
            image_files = [
                f for f in os.listdir(category_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for img_name in image_files:
                img_path = os.path.join(category_path, img_name)
                img = self._load_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(idx)
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images from {len(self.categories)} categories")
        
        return images, labels, self.categories
    
    def load_all_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load all dataset splits (train, validation, test).
        
        Returns:
            Dictionary with keys 'train', 'validation', 'test'
        """
        splits = {}
        
        for split in ['train', 'validation', 'test']:
            try:
                images, labels, _ = self.load_dataset(split)
                splits[split] = (images, labels)
            except FileNotFoundError:
                print(f"Split '{split}' not found, skipping...")
                
        return splits
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Args:
            labels: Array of label indices
            
        Returns:
            Dictionary mapping category names to counts
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {
            self.categories[idx]: count
            for idx, count in zip(unique, counts)
        }
        return distribution
    
    def create_label_map(self) -> Dict[int, str]:
        """
        Create a mapping from label indices to category names.
        
        Returns:
            Dictionary mapping indices to category names
        """
        return {idx: name for idx, name in enumerate(self.categories)}


def load_dataset_simple(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple function to load dataset without class instantiation.
    
    Author: Molla Samser
    Website: https://rskworld.in
    
    Args:
        data_dir: Path to dataset directory
        image_size: Target image size
        
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    categories = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        
        for img_name in os.listdir(category_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, img_name)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    images.append(np.array(img))
                    labels.append(idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # Example usage
    # Author: Molla Samser | https://rskworld.in
    
    print("Image Classification Dataset - Data Loader")
    print("Author: Molla Samser")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("-" * 50)
    
    # Initialize loader
    loader = ImageDataLoader(
        data_dir='../dataset',
        image_size=(224, 224),
        normalize=True
    )
    
    # Load training data
    X_train, y_train, categories = loader.load_dataset('train')
    
    print(f"\nDataset loaded successfully!")
    print(f"Images shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Categories: {categories}")
    
    # Show class distribution
    distribution = loader.get_class_distribution(y_train)
    print(f"\nClass distribution:")
    for category, count in distribution.items():
        print(f"  {category}: {count} images")


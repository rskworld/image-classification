"""
===================================================================================
    Project: Image Classification Dataset
    Description: Image prediction utilities with pre-trained models.
    
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
from typing import Tuple, List, Dict, Optional
import json


class ImagePredictor:
    """
    Image classification predictor with support for multiple frameworks.
    
    Author: Molla Samser
    Website: https://rskworld.in
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
        framework: str = 'tensorflow'
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            class_names_path: Path to JSON file with class names
            framework: 'tensorflow' or 'pytorch'
        """
        self.model = None
        self.class_names = None
        self.framework = framework
        self.image_size = (224, 224)
        
        if model_path:
            self.load_model(model_path)
        
        if class_names_path:
            self.load_class_names(class_names_path)
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        if self.framework == 'tensorflow':
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded from: {model_path}")
            except Exception as e:
                print(f"Error loading TensorFlow model: {e}")
        elif self.framework == 'pytorch':
            try:
                import torch
                self.model = torch.load(model_path)
                self.model.eval()
                print(f"Model loaded from: {model_path}")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
    
    def load_class_names(self, path: str):
        """Load class names from JSON file."""
        with open(path, 'r') as f:
            self.class_names = json.load(f)
        print(f"Loaded {len(self.class_names)} class names")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        
        return img_array
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Predict the class of an image.
        
        Args:
            image_path: Path to the image
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Preprocess
        img = self.preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        if self.framework == 'tensorflow':
            predictions = self.model.predict(img_batch, verbose=0)[0]
        elif self.framework == 'pytorch':
            import torch
            with torch.no_grad():
                img_tensor = torch.tensor(img_batch).permute(0, 3, 1, 2).float()
                predictions = self.model(img_tensor).numpy()[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
            results.append({
                'class_id': int(idx),
                'class_name': class_name,
                'probability': float(predictions[idx]),
                'percentage': f"{predictions[idx] * 100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction results for each image
        """
        return [self.predict(path, top_k) for path in image_paths]
    
    def predict_from_array(self, image_array: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Predict from a numpy array.
        
        Args:
            image_array: Image as numpy array (H, W, C)
            top_k: Number of top predictions
            
        Returns:
            List of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        # Ensure correct shape
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Normalize if needed
        if image_array.max() > 1:
            image_array = image_array / 255.0
        
        predictions = self.model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
            results.append({
                'class_id': int(idx),
                'class_name': class_name,
                'probability': float(predictions[idx])
            })
        
        return results


def predict_image(image_path: str, model_path: str, class_names: List[str]) -> Dict:
    """
    Simple function to predict a single image.
    
    Author: Molla Samser | https://rskworld.in
    
    Args:
        image_path: Path to the image
        model_path: Path to the model
        class_names: List of class names
        
    Returns:
        Prediction result dictionary
    """
    predictor = ImagePredictor(framework='tensorflow')
    predictor.load_model(model_path)
    predictor.class_names = class_names
    
    results = predictor.predict(image_path, top_k=1)
    
    return {
        'image': image_path,
        'predicted_class': results[0]['class_name'],
        'confidence': results[0]['percentage']
    }


class EnsemblePredictor:
    """
    Ensemble prediction using multiple models.
    
    Author: Molla Samser
    Website: https://rskworld.in
    """
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths: List of paths to trained models
            weights: Optional weights for each model
        """
        self.predictors = []
        self.weights = weights or [1.0 / len(model_paths)] * len(model_paths)
        
        for path in model_paths:
            predictor = ImagePredictor(model_path=path)
            self.predictors.append(predictor)
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Make ensemble prediction.
        
        Args:
            image_path: Path to the image
            top_k: Number of top predictions
            
        Returns:
            Ensemble prediction results
        """
        all_predictions = []
        
        for predictor in self.predictors:
            img = predictor.preprocess_image(image_path)
            img_batch = np.expand_dims(img, axis=0)
            pred = predictor.model.predict(img_batch, verbose=0)[0]
            all_predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, self.weights):
            ensemble_pred += pred * weight
        
        # Get top-k
        top_indices = np.argsort(ensemble_pred)[::-1][:top_k]
        
        results = []
        class_names = self.predictors[0].class_names
        
        for idx in top_indices:
            class_name = class_names[idx] if class_names else f"Class {idx}"
            results.append({
                'class_id': int(idx),
                'class_name': class_name,
                'probability': float(ensemble_pred[idx])
            })
        
        return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Image Classification Dataset - Prediction Utilities")
    print("=" * 60)
    print("Author: Molla Samser")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("=" * 60)
    
    print("\nExample usage:")
    print("""
    # Single model prediction
    predictor = ImagePredictor(
        model_path='models/image_classifier.h5',
        class_names_path='models/class_names.json'
    )
    
    results = predictor.predict('test_image.jpg', top_k=5)
    for r in results:
        print(f"{r['class_name']}: {r['percentage']}")
    
    # Ensemble prediction
    ensemble = EnsemblePredictor([
        'models/model1.h5',
        'models/model2.h5',
        'models/model3.h5'
    ])
    
    results = ensemble.predict('test_image.jpg')
    """)


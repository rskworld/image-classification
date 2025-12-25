"""
===================================================================================
    Project: Image Classification Dataset
    Description: Python scripts for image classification tasks.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
"""

from .data_loader import ImageDataLoader, load_dataset_simple
from .train_model import ImageClassificationModel
from .augmentation import ImageAugmentor, augment_and_save
from .evaluate import ModelEvaluator
from .advanced_utils import DatasetAnalyzer, GradCAMVisualizer, DatasetSplitter
from .predict import ImagePredictor, EnsemblePredictor, predict_image

__all__ = [
    # Data Loading
    'ImageDataLoader',
    'load_dataset_simple',
    # Model Training
    'ImageClassificationModel',
    # Augmentation
    'ImageAugmentor',
    'augment_and_save',
    # Evaluation
    'ModelEvaluator',
    # Advanced Utilities
    'DatasetAnalyzer',
    'GradCAMVisualizer',
    'DatasetSplitter',
    # Prediction
    'ImagePredictor',
    'EnsemblePredictor',
    'predict_image'
]

__version__ = '1.0.0'
__author__ = 'Molla Samser'
__email__ = 'help@rskworld.in'
__website__ = 'https://rskworld.in'


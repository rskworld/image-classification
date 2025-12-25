"""
===================================================================================
    Project: Image Classification Dataset
    Description: Advanced utilities for image classification tasks.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable
from collections import Counter
import random
from pathlib import Path


class DatasetAnalyzer:
    """
    Comprehensive dataset analysis and visualization toolkit.
    
    Author: Molla Samser
    Website: https://rskworld.in
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            dataset_dir: Path to the dataset root directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Optional[Dict]:
        """Load dataset metadata if available."""
        metadata_path = self.dataset_dir.parent / 'data' / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_dataset_stats(self) -> Dict:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'splits': {},
            'total_images': 0,
            'categories': set(),
            'file_formats': Counter(),
            'image_sizes': []
        }
        
        for split in ['train', 'validation', 'test']:
            split_path = self.dataset_dir / split
            if not split_path.exists():
                continue
                
            split_stats = {'categories': {}, 'total': 0}
            
            for category in split_path.iterdir():
                if category.is_dir():
                    images = list(category.glob('*.*'))
                    count = len([f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    split_stats['categories'][category.name] = count
                    split_stats['total'] += count
                    stats['categories'].add(category.name)
                    
                    # Sample file formats
                    for img in images[:10]:
                        stats['file_formats'][img.suffix.lower()] += 1
            
            stats['splits'][split] = split_stats
            stats['total_images'] += split_stats['total']
        
        stats['categories'] = sorted(list(stats['categories']))
        stats['file_formats'] = dict(stats['file_formats'])
        
        return stats
    
    def analyze_image_properties(self, sample_size: int = 100) -> Dict:
        """
        Analyze image properties (size, channels, etc.).
        
        Args:
            sample_size: Number of images to sample
            
        Returns:
            Dictionary with image property statistics
        """
        properties = {
            'widths': [],
            'heights': [],
            'channels': [],
            'aspect_ratios': [],
            'file_sizes_kb': []
        }
        
        # Collect sample images
        all_images = []
        for split_path in self.dataset_dir.iterdir():
            if split_path.is_dir():
                for category in split_path.iterdir():
                    if category.is_dir():
                        all_images.extend(list(category.glob('*.jpg')) + list(category.glob('*.png')))
        
        # Sample
        sample = random.sample(all_images, min(sample_size, len(all_images)))
        
        for img_path in sample:
            try:
                img = Image.open(img_path)
                w, h = img.size
                properties['widths'].append(w)
                properties['heights'].append(h)
                properties['aspect_ratios'].append(w / h)
                properties['channels'].append(len(img.getbands()))
                properties['file_sizes_kb'].append(os.path.getsize(img_path) / 1024)
            except Exception as e:
                continue
        
        # Calculate statistics
        return {
            'avg_width': np.mean(properties['widths']) if properties['widths'] else 0,
            'avg_height': np.mean(properties['heights']) if properties['heights'] else 0,
            'avg_aspect_ratio': np.mean(properties['aspect_ratios']) if properties['aspect_ratios'] else 0,
            'common_channels': Counter(properties['channels']).most_common(1)[0] if properties['channels'] else None,
            'avg_file_size_kb': np.mean(properties['file_sizes_kb']) if properties['file_sizes_kb'] else 0,
            'min_size': (min(properties['widths']), min(properties['heights'])) if properties['widths'] else (0, 0),
            'max_size': (max(properties['widths']), max(properties['heights'])) if properties['widths'] else (0, 0)
        }
    
    def plot_category_distribution(self, split: str = 'train', save_path: Optional[str] = None):
        """
        Plot category distribution as a bar chart.
        
        Args:
            split: Which split to analyze
            save_path: Optional path to save the plot
        """
        stats = self.get_dataset_stats()
        
        if split not in stats['splits']:
            print(f"Split '{split}' not found")
            return
        
        categories = list(stats['splits'][split]['categories'].keys())
        counts = list(stats['splits'][split]['categories'].values())
        
        # Sort by count
        sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
        categories, counts = zip(*sorted_data)
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(categories, counts, color=plt.cm.viridis(np.linspace(0, 0.8, len(categories))))
        
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title(f'Category Distribution ({split.capitalize()} Set)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_split_distribution(self, save_path: Optional[str] = None):
        """
        Plot data split distribution (train/val/test).
        
        Args:
            save_path: Optional path to save the plot
        """
        stats = self.get_dataset_stats()
        
        splits = []
        totals = []
        
        for split, data in stats['splits'].items():
            splits.append(split.capitalize())
            totals.append(data['total'])
        
        colors = ['#00d4ff', '#7b68ee', '#ff6b6b']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        bars = ax1.bar(splits, totals, color=colors)
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Split Distribution (Bar)')
        
        for bar, total in zip(bars, totals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(total), ha='center', fontsize=10)
        
        # Pie chart
        ax2.pie(totals, labels=splits, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=[0.02] * len(splits))
        ax2.set_title('Split Distribution (Percentage)')
        
        plt.suptitle('Dataset Split Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir: str = 'analysis_report'):
        """
        Generate a comprehensive dataset analysis report.
        
        Args:
            output_dir: Directory to save the report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("IMAGE CLASSIFICATION DATASET - ANALYSIS REPORT")
        print("Author: Molla Samser | https://rskworld.in")
        print("=" * 60)
        
        # Get statistics
        stats = self.get_dataset_stats()
        properties = self.analyze_image_properties()
        
        # Print summary
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Images: {stats['total_images']:,}")
        print(f"   Categories: {len(stats['categories'])}")
        print(f"   File Formats: {stats['file_formats']}")
        
        print(f"\n📐 Image Properties:")
        print(f"   Average Size: {properties['avg_width']:.0f} × {properties['avg_height']:.0f}")
        print(f"   Aspect Ratio: {properties['avg_aspect_ratio']:.2f}")
        print(f"   Avg File Size: {properties['avg_file_size_kb']:.1f} KB")
        
        print(f"\n📁 Split Distribution:")
        for split, data in stats['splits'].items():
            print(f"   {split.capitalize()}: {data['total']:,} images")
        
        # Generate plots
        print("\n📈 Generating visualizations...")
        self.plot_category_distribution('train', os.path.join(output_dir, 'category_distribution.png'))
        self.plot_split_distribution(os.path.join(output_dir, 'split_distribution.png'))
        
        # Save JSON report
        report = {
            'dataset_stats': stats,
            'image_properties': properties,
            'metadata': self.metadata
        }
        
        with open(os.path.join(output_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Report generated in: {output_dir}")


class GradCAMVisualizer:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability.
    
    Author: Molla Samser
    Website: https://rskworld.in
    """
    
    def __init__(self, model, target_layer_name: str):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: Trained Keras/TensorFlow model
            target_layer_name: Name of the target convolutional layer
        """
        self.model = model
        self.target_layer_name = target_layer_name
        
    def compute_gradcam(self, image: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image: Input image array
            class_idx: Target class index
            
        Returns:
            Heatmap array
        """
        try:
            import tensorflow as tf
            
            # Create gradient model
            grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[
                    self.model.get_layer(self.target_layer_name).output,
                    self.model.output
                ]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(image[np.newaxis, ...])
                loss = predictions[:, class_idx]
            
            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_output = conv_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.nn.relu(heatmap)
            heatmap = heatmap / tf.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except ImportError:
            print("TensorFlow required for Grad-CAM visualization")
            return None
    
    def visualize(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            alpha: Overlay transparency
            
        Returns:
            Superimposed visualization
        """
        import cv2
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed


class DatasetSplitter:
    """
    Utility for splitting datasets into train/validation/test sets.
    
    Author: Molla Samser
    Website: https://rskworld.in
    """
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize the splitter.
        
        Args:
            source_dir: Directory containing all images organized by category
            output_dir: Directory to save the split dataset
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Split the dataset.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directories
        for split in ['train', 'validation', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        print(f"Splitting dataset: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
        
        for category in self.source_dir.iterdir():
            if not category.is_dir():
                continue
                
            # Get all images
            images = list(category.glob('*.jpg')) + list(category.glob('*.png')) + list(category.glob('*.jpeg'))
            random.shuffle(images)
            
            n = len(images)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                'train': images[:train_end],
                'validation': images[train_end:val_end],
                'test': images[val_end:]
            }
            
            # Copy files
            for split_name, split_images in splits.items():
                split_category_dir = self.output_dir / split_name / category.name
                split_category_dir.mkdir(parents=True, exist_ok=True)
                
                for img in split_images:
                    import shutil
                    shutil.copy2(img, split_category_dir / img.name)
            
            print(f"  {category.name}: {len(splits['train'])} train, {len(splits['validation'])} val, {len(splits['test'])} test")
        
        print("\n✅ Dataset split complete!")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Image Classification Dataset - Advanced Utilities")
    print("=" * 60)
    print("Author: Molla Samser")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("=" * 60)
    
    print("\nAvailable classes:")
    print("  - DatasetAnalyzer: Comprehensive dataset analysis")
    print("  - GradCAMVisualizer: Model interpretability with Grad-CAM")
    print("  - DatasetSplitter: Split raw datasets into train/val/test")
    
    print("\nExample usage:")
    print("  analyzer = DatasetAnalyzer('../dataset')")
    print("  analyzer.generate_report()")


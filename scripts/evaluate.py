"""
===================================================================================
    Project: Image Classification Dataset
    Description: Model evaluation and visualization utilities.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from typing import List, Optional, Tuple, Dict
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class/category names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute various classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Print overall metrics
        metrics = self.compute_metrics(y_true, y_pred)
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print("=" * 60)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            save_path: Optional path to save the plot
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_class_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot per-class precision, recall, and F1 scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the plot
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#00d4ff')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#7b68ee')
        bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#ff6b6b')
        
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Class performance plot saved to: {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(
        self,
        images: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_samples: int = 16,
        save_path: Optional[str] = None
    ):
        """
        Plot sample images with predictions.
        
        Args:
            images: Array of images
            y_true: True labels
            y_pred: Predicted labels
            num_samples: Number of samples to show
            save_path: Optional path to save the plot
        """
        # Find correct and incorrect predictions
        correct_mask = y_true == y_pred
        incorrect_indices = np.where(~correct_mask)[0]
        
        # Prioritize showing misclassifications
        if len(incorrect_indices) >= num_samples:
            indices = np.random.choice(incorrect_indices, num_samples, replace=False)
        else:
            # Mix of correct and incorrect
            correct_indices = np.where(correct_mask)[0]
            n_incorrect = len(incorrect_indices)
            n_correct = num_samples - n_incorrect
            indices = np.concatenate([
                incorrect_indices,
                np.random.choice(correct_indices, n_correct, replace=False)
            ])
        
        # Plot
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            # Handle normalized images
            img = images[idx]
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            
            ax.imshow(img)
            
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            is_correct = y_true[idx] == y_pred[idx]
            color = 'green' if is_correct else 'red'
            
            ax.set_title(
                f'True: {true_label}\nPred: {pred_label}',
                color=color,
                fontsize=10
            )
            ax.axis('off')
        
        # Hide unused axes
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample predictions saved to: {save_path}")
        
        plt.show()
    
    def plot_accuracy_by_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot accuracy for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Sort by accuracy
        sorted_indices = np.argsort(class_accuracy)
        sorted_names = [self.class_names[i] for i in sorted_indices]
        sorted_accuracy = class_accuracy[sorted_indices]
        
        # Create color gradient
        colors = plt.cm.RdYlGn(sorted_accuracy)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_names)), sorted_accuracy, color=colors)
        
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Accuracy', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xlim(0, 1.05)
        
        # Add value labels
        for bar, acc in zip(bars, sorted_accuracy):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2%}', va='center', fontsize=9)
        
        plt.axvline(x=np.mean(class_accuracy), color='blue', linestyle='--',
                   label=f'Mean: {np.mean(class_accuracy):.2%}')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Accuracy by class plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        images: Optional[np.ndarray] = None,
        output_dir: str = 'evaluation_results'
    ):
        """
        Generate complete evaluation report with all visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            images: Optional image array for sample visualization
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("GENERATING EVALUATION REPORT")
        print("Author: Molla Samser | https://rskworld.in")
        print("=" * 60)
        
        # Print classification report
        self.print_classification_report(y_true, y_pred)
        
        # Generate visualizations
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        self.plot_class_performance(
            y_true, y_pred,
            save_path=os.path.join(output_dir, 'class_performance.png')
        )
        
        self.plot_accuracy_by_class(
            y_true, y_pred,
            save_path=os.path.join(output_dir, 'accuracy_by_class.png')
        )
        
        if images is not None:
            self.plot_sample_predictions(
                images, y_true, y_pred,
                save_path=os.path.join(output_dir, 'sample_predictions.png')
            )
        
        # Save metrics to file
        metrics = self.compute_metrics(y_true, y_pred)
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write("Image Classification Dataset - Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write("Author: Molla Samser | https://rskworld.in\n")
            f.write("Email: help@rskworld.in\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Metrics:\n")
            for name, value in metrics.items():
                f.write(f"  {name}: {value:.4f}\n")
            
            f.write("\n" + classification_report(y_true, y_pred, target_names=self.class_names))
        
        print(f"\nReport saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    # Author: Molla Samser | https://rskworld.in
    
    print("=" * 60)
    print("Image Classification Dataset - Model Evaluation")
    print("=" * 60)
    print("Author: Molla Samser")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("=" * 60)
    
    # Example with dummy data
    class_names = ['Animals', 'Vehicles', 'Nature', 'Food', 'Buildings']
    
    evaluator = ModelEvaluator(class_names)
    
    print("\nExample usage:")
    print("  evaluator = ModelEvaluator(class_names)")
    print("  evaluator.generate_report(y_true, y_pred, images)")
    
    print("\nAvailable methods:")
    print("  - compute_metrics(): Compute accuracy, precision, recall, F1")
    print("  - print_classification_report(): Print detailed report")
    print("  - plot_confusion_matrix(): Visualize confusion matrix")
    print("  - plot_class_performance(): Bar chart of per-class metrics")
    print("  - plot_sample_predictions(): Show sample images with predictions")
    print("  - plot_accuracy_by_class(): Horizontal bar chart of accuracies")
    print("  - generate_report(): Generate complete evaluation report")


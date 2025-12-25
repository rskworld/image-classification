"""
===================================================================================
    Project: Image Classification Dataset
    Description: CNN model training script for image classification.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime


class ImageClassificationModel:
    """
    CNN model for image classification tasks.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
        model_name: str = 'cnn_classifier'
    ):
        """
        Initialize the model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classification categories
            model_name: Name for the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        
    def build_cnn_model(self) -> keras.Model:
        """
        Build a custom CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_transfer_learning_model(
        self,
        base_model: str = 'resnet50',
        trainable_layers: int = 20
    ) -> keras.Model:
        """
        Build a transfer learning model using pre-trained weights.
        
        Args:
            base_model: Name of the pre-trained model ('resnet50', 'vgg16', 'efficientnet')
            trainable_layers: Number of layers to make trainable
            
        Returns:
            Compiled Keras model
        """
        # Select base model
        if base_model == 'resnet50':
            base = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model == 'vgg16':
            base = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model == 'efficientnet':
            base = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Freeze early layers
        for layer in base.layers[:-trainable_layers]:
            layer.trainable = False
        
        # Build complete model
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def create_data_generator(
        self,
        augment: bool = True
    ) -> ImageDataGenerator:
        """
        Create an image data generator with optional augmentation.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            ImageDataGenerator instance
        """
        if augment:
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator(rescale=1./255)
            
        return datagen
    
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        augment: bool = True,
        early_stopping: bool = True,
        save_best: bool = True
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_data: Tuple of (images, labels)
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size
            augment: Whether to use data augmentation
            early_stopping: Whether to use early stopping
            save_best: Whether to save the best model
            
        Returns:
            Training history
        """
        X_train, y_train = train_data
        
        # Convert labels to one-hot encoding
        y_train_onehot = keras.utils.to_categorical(y_train, self.num_classes)
        
        if val_data:
            X_val, y_val = val_data
            y_val_onehot = keras.utils.to_categorical(y_val, self.num_classes)
            validation_data = (X_val, y_val_onehot)
        else:
            validation_data = None
        
        # Create callbacks
        callback_list = []
        
        if early_stopping:
            callback_list.append(callbacks.EarlyStopping(
                monitor='val_loss' if val_data else 'loss',
                patience=10,
                restore_best_weights=True
            ))
        
        if save_best:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'models/{self.model_name}_{timestamp}.h5'
            os.makedirs('models', exist_ok=True)
            callback_list.append(callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy' if val_data else 'accuracy',
                save_best_only=True
            ))
        
        # Add learning rate scheduler
        callback_list.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ))
        
        # Train with or without augmentation
        if augment:
            datagen = self.create_data_generator(augment=True)
            
            self.history = self.model.fit(
                datagen.flow(X_train, y_train_onehot, batch_size=batch_size),
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callback_list,
                steps_per_epoch=len(X_train) // batch_size
            )
        else:
            self.history = self.model.fit(
                X_train, y_train_onehot,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callback_list
            )
        
        return self.history
    
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Tuple of (images, labels)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        X_test, y_test = test_data
        y_test_onehot = keras.utils.to_categorical(y_test, self.num_classes)
        
        loss, accuracy = self.model.evaluate(X_test, y_test_onehot)
        
        print(f"\nTest Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return loss, accuracy
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Make predictions on new images.
        
        Args:
            images: Array of images
            
        Returns:
            Predicted class indices
        """
        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        if 'val_accuracy' in self.history.history:
            axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train')
        if 'val_loss' in self.history.history:
            axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path: str):
        """Save the model to disk."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a model from disk."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    # Author: Molla Samser | https://rskworld.in
    
    print("=" * 60)
    print("Image Classification Dataset - Model Training")
    print("=" * 60)
    print("Author: Molla Samser")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("=" * 60)
    
    # Example: Create and train a model
    classifier = ImageClassificationModel(
        input_shape=(224, 224, 3),
        num_classes=12,
        model_name='image_classifier'
    )
    
    # Build CNN model
    model = classifier.build_cnn_model()
    model.summary()
    
    print("\nModel built successfully!")
    print("To train the model, use the train() method with your data.")
    print("\nExample:")
    print("  classifier.train(train_data=(X_train, y_train), epochs=50)")


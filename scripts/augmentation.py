"""
===================================================================================
    Project: Image Classification Dataset
    Description: Image augmentation utilities for expanding training data.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, List, Optional, Callable
import random
from tqdm import tqdm


class ImageAugmentor:
    """
    Comprehensive image augmentation class for data expansion.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the augmentor.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # ==================== Basic Transformations ====================
    
    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically."""
        return cv2.flip(image, 0)
    
    def zoom(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Zoom into image.
        
        Args:
            image: Input image
            factor: Zoom factor (1.0 = no zoom, >1.0 = zoom in)
            
        Returns:
            Zoomed image
        """
        h, w = image.shape[:2]
        
        # Calculate crop dimensions
        new_h = int(h / factor)
        new_w = int(w / factor)
        
        # Calculate crop coordinates
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        
        # Crop and resize
        cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    def shift(
        self,
        image: np.ndarray,
        x_shift: float,
        y_shift: float
    ) -> np.ndarray:
        """
        Shift image by specified amounts.
        
        Args:
            image: Input image
            x_shift: Horizontal shift as fraction of width
            y_shift: Vertical shift as fraction of height
            
        Returns:
            Shifted image
        """
        h, w = image.shape[:2]
        dx = int(w * x_shift)
        dy = int(h * y_shift)
        
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return shifted
    
    def shear(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply shear transformation.
        
        Args:
            image: Input image
            factor: Shear factor
            
        Returns:
            Sheared image
        """
        h, w = image.shape[:2]
        matrix = np.float32([[1, factor, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return sheared
    
    # ==================== Color Transformations ====================
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (1.0 = no change)
            
        Returns:
            Brightness-adjusted image
        """
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)
            
        Returns:
            Contrast-adjusted image
        """
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image saturation.
        
        Args:
            image: Input image
            factor: Saturation factor (1.0 = no change)
            
        Returns:
            Saturation-adjusted image
        """
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def adjust_hue(self, image: np.ndarray, shift: float) -> np.ndarray:
        """
        Shift image hue.
        
        Args:
            image: Input image
            shift: Hue shift value (-180 to 180)
            
        Returns:
            Hue-shifted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + int(shift)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # ==================== Noise & Filter Transformations ====================
    
    def add_gaussian_noise(
        self,
        image: np.ndarray,
        mean: float = 0,
        std: float = 25
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image
            mean: Noise mean
            std: Noise standard deviation
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur.
        
        Args:
            image: Input image
            kernel_size: Blur kernel size
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    # ==================== Random Augmentation ====================
    
    def random_augment(
        self,
        image: np.ndarray,
        augmentations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image
            augmentations: List of augmentations to apply randomly
            
        Returns:
            Augmented image
        """
        if augmentations is None:
            augmentations = [
                'rotate', 'flip_h', 'flip_v', 'zoom', 'shift',
                'brightness', 'contrast', 'saturation', 'noise', 'blur'
            ]
        
        result = image.copy()
        
        # Randomly select augmentations to apply
        num_augments = random.randint(1, 3)
        selected = random.sample(augmentations, min(num_augments, len(augmentations)))
        
        for aug in selected:
            if aug == 'rotate':
                angle = random.uniform(-30, 30)
                result = self.rotate(result, angle)
            elif aug == 'flip_h' and random.random() > 0.5:
                result = self.flip_horizontal(result)
            elif aug == 'flip_v' and random.random() > 0.5:
                result = self.flip_vertical(result)
            elif aug == 'zoom':
                factor = random.uniform(1.0, 1.3)
                result = self.zoom(result, factor)
            elif aug == 'shift':
                x_shift = random.uniform(-0.1, 0.1)
                y_shift = random.uniform(-0.1, 0.1)
                result = self.shift(result, x_shift, y_shift)
            elif aug == 'brightness':
                factor = random.uniform(0.7, 1.3)
                result = self.adjust_brightness(result, factor)
            elif aug == 'contrast':
                factor = random.uniform(0.7, 1.3)
                result = self.adjust_contrast(result, factor)
            elif aug == 'saturation':
                factor = random.uniform(0.7, 1.3)
                result = self.adjust_saturation(result, factor)
            elif aug == 'noise':
                result = self.add_gaussian_noise(result, std=random.uniform(5, 20))
            elif aug == 'blur':
                if random.random() > 0.7:
                    result = self.apply_blur(result, kernel_size=3)
        
        return result
    
    def augment_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augmentations_per_image: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment entire dataset.
        
        Args:
            images: Array of images
            labels: Array of labels
            augmentations_per_image: Number of augmented versions per image
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        print(f"Augmenting {len(images)} images...")
        
        for i, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images))):
            # Add original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augmentations_per_image):
                aug_img = self.random_augment(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)


def augment_and_save(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5
):
    """
    Augment images and save to new directory.
    
    Author: Molla Samser
    Website: https://rskworld.in
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        augmentations_per_image: Number of augmented versions per image
    """
    augmentor = ImageAugmentor(seed=42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for category in os.listdir(input_dir):
        category_input = os.path.join(input_dir, category)
        category_output = os.path.join(output_dir, category)
        
        if not os.path.isdir(category_input):
            continue
            
        os.makedirs(category_output, exist_ok=True)
        
        print(f"\nProcessing category: {category}")
        
        for img_name in tqdm(os.listdir(category_input)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(category_input, img_name)
            
            try:
                img = np.array(Image.open(img_path).convert('RGB'))
                
                # Save original
                base_name = os.path.splitext(img_name)[0]
                Image.fromarray(img).save(
                    os.path.join(category_output, f"{base_name}_orig.jpg")
                )
                
                # Generate and save augmented versions
                for i in range(augmentations_per_image):
                    aug_img = augmentor.random_augment(img)
                    Image.fromarray(aug_img).save(
                        os.path.join(category_output, f"{base_name}_aug{i+1}.jpg")
                    )
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"\nAugmentation complete! Output saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    # Author: Molla Samser | https://rskworld.in
    
    print("=" * 60)
    print("Image Classification Dataset - Augmentation")
    print("=" * 60)
    print("Author: Molla Samser")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("=" * 60)
    
    # Example: Augment a single image
    augmentor = ImageAugmentor(seed=42)
    
    print("\nAvailable augmentations:")
    print("  - rotate: Random rotation")
    print("  - flip_h: Horizontal flip")
    print("  - flip_v: Vertical flip")
    print("  - zoom: Random zoom")
    print("  - shift: Random shift")
    print("  - brightness: Brightness adjustment")
    print("  - contrast: Contrast adjustment")
    print("  - saturation: Saturation adjustment")
    print("  - noise: Gaussian noise")
    print("  - blur: Gaussian blur")
    
    print("\nTo augment a dataset directory, use:")
    print("  augment_and_save('input_dir', 'output_dir', augmentations_per_image=5)")


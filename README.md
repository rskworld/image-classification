<!--
===================================================================================
    Project: Image Classification Dataset
    Description: Large image classification dataset with multiple categories 
                 for training CNN models, transfer learning, and image recognition tasks.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
-->

# 🖼️ Image Classification Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

> Large image classification dataset with multiple categories for training CNN models, transfer learning, and image recognition tasks.

## 📋 Overview

This dataset contains thousands of labeled images across multiple categories, organized for image classification tasks. Perfect for training CNN models, transfer learning, image recognition, and computer vision applications.

## ✨ Features

- 🖼️ **Multiple Image Categories** - Diverse categories covering animals, vehicles, nature, objects, and more
- 🏷️ **Labeled Training Data** - All images properly labeled and organized by category
- 📊 **Test & Validation Sets** - Pre-split into training (70%), validation (15%), and test (15%) sets
- 📐 **Various Image Sizes** - Images in multiple resolutions for different model requirements
- 🧠 **Ready for CNN Training** - Optimized format for Convolutional Neural Networks

## 📁 Dataset Structure

```
image-classification/
├── dataset/
│   ├── train/              # 70% of data
│   │   ├── animals/
│   │   ├── vehicles/
│   │   ├── nature/
│   │   ├── food/
│   │   ├── buildings/
│   │   └── ...
│   ├── validation/         # 15% of data
│   │   ├── animals/
│   │   ├── vehicles/
│   │   └── ...
│   └── test/               # 15% of data
│       ├── animals/
│       ├── vehicles/
│       └── ...
├── scripts/
│   ├── data_loader.py
│   ├── train_model.py
│   ├── augmentation.py
│   └── evaluate.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-classification.git
cd image-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Image Classification Dataset - rskworld.in
# Author: Molla Samser | Email: help@rskworld.in

import os
import numpy as np
from PIL import Image

def load_dataset(data_dir):
    """Load images from the dataset directory."""
    images = []
    labels = []
    categories = os.listdir(data_dir)
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).resize((224, 224))
            images.append(np.array(img))
            labels.append(idx)
    
    return np.array(images), np.array(labels)

# Load training data
X_train, y_train = load_dataset('dataset/train')
print(f"Training samples: {len(X_train)}")
```

## 🛠️ Technologies

| Technology | Description |
|------------|-------------|
| **PNG/JPG** | Image formats used in the dataset |
| **NumPy** | Numerical computing for array operations |
| **PIL/Pillow** | Python Imaging Library for image processing |
| **OpenCV** | Computer vision library for advanced operations |
| **TensorFlow** | Deep learning framework |
| **PyTorch** | Deep learning framework |

## 📊 Dataset Statistics

| Category | Training | Validation | Test |
|----------|----------|------------|------|
| Animals | 1,050 | 225 | 225 |
| Vehicles | 840 | 180 | 180 |
| Nature | 700 | 150 | 150 |
| Food | 560 | 120 | 120 |
| Buildings | 630 | 135 | 135 |
| Fashion | 490 | 105 | 105 |
| Aircraft | 420 | 90 | 90 |
| Sports | 560 | 120 | 120 |
| Instruments | 350 | 75 | 75 |
| Electronics | 490 | 105 | 105 |
| Furniture | 420 | 90 | 90 |
| Plants | 490 | 105 | 105 |
| **Total** | **7,000** | **1,500** | **1,500** |

## 🧠 Model Training Examples

### TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### PyTorch

```python
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## 📈 Data Augmentation

The dataset includes augmentation scripts for expanding your training data:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Molla Samser**

- 🌐 Website: [rskworld.in](https://rskworld.in)
- 📧 Email: [help@rskworld.in](mailto:help@rskworld.in)
- 📱 Phone: +91 93305 39277

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/image-classification/issues).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

## 📞 Contact

For any inquiries or support, please reach out:

- **Email:** help@rskworld.in
- **Phone:** +91 93305 39277
- **Website:** [https://rskworld.in](https://rskworld.in)
- **Contact Page:** [https://rskworld.in/contact.php](https://rskworld.in/contact.php)

---

<p align="center">
  Made with ❤️ by <a href="https://rskworld.in">RSK World</a>
</p>


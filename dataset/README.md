<!--
===================================================================================
    Project: Image Classification Dataset
    Description: Dataset directory structure guide.
    
    Author: Molla Samser
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Website: https://rskworld.in
    
    © 2025 RSK World. All rights reserved.
===================================================================================
-->

# 📁 Dataset Directory Structure

This directory should contain your image classification dataset organized as follows:

```
dataset/
├── train/                  # Training data (70%)
│   ├── animals/
│   │   ├── cat_001.jpg
│   │   ├── dog_001.jpg
│   │   └── ...
│   ├── vehicles/
│   │   ├── car_001.jpg
│   │   └── ...
│   └── [other_categories]/
│
├── validation/             # Validation data (15%)
│   ├── animals/
│   ├── vehicles/
│   └── [other_categories]/
│
└── test/                   # Test data (15%)
    ├── animals/
    ├── vehicles/
    └── [other_categories]/
```

## 📋 Guidelines

1. **Consistent Structure**: Ensure all three splits (train, validation, test) have the same category directories.

2. **Naming Convention**: Use descriptive names for categories (e.g., `animals`, not `cat1`).

3. **Image Formats**: Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

4. **Image Quality**: Use high-quality, well-lit images for better model performance.

5. **Class Balance**: Try to maintain similar numbers of images per category.

## 🔧 Adding Your Data

1. Create category subdirectories in `train/`, `validation/`, and `test/`
2. Place corresponding images in each category
3. Maintain approximately 70-15-15 split ratio

## 📞 Support

For any questions or assistance:
- **Email:** help@rskworld.in
- **Website:** https://rskworld.in

---
*Author: Molla Samser | RSK World*


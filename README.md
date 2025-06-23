# Satellite Image-Based Crop Health Detection

This project aims to classify crop health (Healthy, Rust, Other) using NDVI satellite images and a CNN built with Keras.

## 📂 Folder Structure
- `data/raw` - Unprocessed NDVI images.
- `data/processed` - Preprocessed/resized images for training.
- `notebooks` - For prototyping and EDA.
- `scripts` - Reusable code (training, labeling).
- `models` - Trained Keras models.
- `outputs/plots` - Accuracy curves, confusion matrix, etc.
- `outputs/reports` - Final report files.

## 🚀 Goals
- Train a CNN model using NDVI images.
- Classify crops into 3 categories: Healthy, Rust, Other.
- Help farmers detect early crop stress using satellite data.

## ✅ Tools Used
- Python, TensorFlow/Keras
- Google Earth Engine (for NDVI)
- OpenCV, NumPy, Matplotlib

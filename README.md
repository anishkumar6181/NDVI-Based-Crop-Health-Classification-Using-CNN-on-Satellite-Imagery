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

## 📊 Model Results & Observations
- The current CNN model is able to fit the training data but shows signs of **overfitting**: validation accuracy remains low and the model is biased toward predicting the 'Rust' class.
- Example validation results (see `outputs/predictions/validation_predictions.csv`) show many misclassifications, especially for the 'Health' class.
- This suggests the model is not generalizing well to unseen data.

## ⚠️ Known Issues & Recommendations
- **Overfitting**: The model performs well on training data but poorly on validation data.
  - **Solutions:**
    - Use EarlyStopping during training to halt when validation loss stops improving.
    - Apply data augmentation to increase dataset diversity.
    - Consider simplifying the CNN architecture or increasing dropout.
- **FileNotFoundError**: If you see errors about missing files or folders, ensure you are using the correct absolute paths as set up in the notebook's configuration cell. Do not use relative paths from the notebook directory.

## 🚀 Goals
- Train a CNN model using NDVI images.
- Classify crops into 3 categories: Healthy, Rust, Other.
- Help farmers detect early crop stress using satellite data.

## ✅ Tools Used
- Python, TensorFlow/Keras
- Google Earth Engine (for NDVI)
- OpenCV, NumPy, Matplotlib

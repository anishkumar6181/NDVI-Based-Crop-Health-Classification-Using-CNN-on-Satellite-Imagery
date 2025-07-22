from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf # Import tensorflow

def build_cnn_model(input_shape=(64, 64, 1), num_classes=3):
    """
    Enhanced CNN model for crop health classification with:
    - Better handling of class imbalance
    - Reduced overfitting
    - Improved feature extraction
    
    Args:
        input_shape: Input image shape (default: 64x64 grayscale)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', activation='swish', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))  # Early regularization

    # Block 2 
    model.add(Conv2D(64, (3, 3), padding='same', activation='swish'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='swish'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())  # Better than Flatten for small datasets
    model.add(Dropout(0.4))

    # Classifier
    model.add(Dense(128, activation='swish', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision')]
    )
    
    return model
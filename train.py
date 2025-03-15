import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, SeparableConv2D, BatchNormalization, Activation,
                                     ZeroPadding2D, Add, AveragePooling2D, MaxPooling2D, Concatenate,
                                     GlobalAveragePooling2D, Dense, Dropout)

def create_model():
    inputs = Input(shape=(224, 224, 3))
    
    # Initial Conv Layer
    x = Conv2D(32, (3, 3), padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Example Separable Convolution Block
    x = SeparableConv2D(64, (3, 3), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Global Pooling and Dense Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)  # Adjust classes as needed
    
    model = Model(inputs, x)
    return model

# Create and save the model
model = create_model()
model.save("model2.h5")

# Part 1: Training the Organ Classifier

# Import necessary libraries
import os
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import nibabel as nib  # Library to handle .nii and .nii.gz files
import pydicom  # Library to handle .dcm files
from tensorflow.keras.uti ls import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile  # Handle image loading and truncated images

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define constants for organ classification
data_dir = "dataset"  # Path to organ dataset
categories = ["brain", "heart", "breast", "limbs", "unknown"]  # Organ categories
img_size = 224  # EfficientNetB0 requires 224x224 images

# Use ImageDataGenerator to handle large datasets without loading everything into memory
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Rescale images and split for validation

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Step 1: Load EfficientNetB0 and Add Custom Layers for Organ Classification
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling to reduce the spatial dimensions
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(5, activation='softmax')(x)  # Output layer

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile and Train the Organ Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Fine-Tune the Model
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers for fine-tuning
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)

model.save('fine_tuned_organ_classification_model.keras')



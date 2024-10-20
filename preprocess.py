import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
img_size = 224  # Size to which each image will be resized (224x224 pixels)
batch_size = 32  # Batch size for training

# Create an instance of ImageDataGenerator for preprocessing with augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # Rotate images
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    shear_range=0.2,  # Shearing
    zoom_range=0.2,  # Zoom
    horizontal_flip=True  # Flip horizontally
)


# Corrected paths
train_dir = 'C:/Users/singh/Downloads/ewastedetection/archive/modified-dataset/train'
val_dir = 'C:/Users/singh/Downloads/ewastedetection/archive/modified-dataset/val'

# Load and preprocess the training dataset
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),  # Resize images to 224x224
    batch_size=batch_size,
    class_mode='categorical'  # Multiclass classification
)

# Load and preprocess the validation dataset
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

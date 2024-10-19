import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
img_size = 224  # Size to which each image will be resized (224x224 pixels)
batch_size = 32  # Batch size for training

# Create an instance of ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values (scale them to 0-1)
    validation_split=0.2  # Split 20% of data for validation
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

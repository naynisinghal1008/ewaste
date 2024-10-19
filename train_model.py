import tensorflow as tf
from tensorflow.keras import layers, models

# Build a simple CNN model
model = models.Sequential()

# Add Convolutional layers and MaxPooling layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add Fully Connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Adjust the number of classes if different

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
# Load the training and validation data from preprocess.py
from preprocess import train_generator, val_generator

# Train the CNN model
history = model.fit(
    train_generator,
    epochs=10,  # You can change the number of epochs
    validation_data=val_generator
)
# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
model.save('ewaste_detection_model.h5')
import tensorflow as tf
from tensorflow.keras import layers, models

# Build a deeper CNN model
model = models.Sequential()

# Add Convolutional layers
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add more Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Add dropout to prevent overfitting
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Print the model summary
model.summary()

# Load the training and validation data from preprocess.py
from preprocess import train_generator, val_generator

# Train the CNN model
history = model.fit(
    train_generator,
    epochs=30,  # You can change the number of epochs
    validation_data=val_generator
)
# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
model.save('ewaste_detection_model.keras')


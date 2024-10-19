import cv2  # For accessing the webcam
import numpy as np
import tensorflow as tf  # To load the trained model

# Load the trained model
model = tf.keras.models.load_model('ewaste_detection_model.h5')

# Define image size
img_size = 224  # This should match the input size your model expects (224x224)

# Define a function to preprocess the frame captured by the camera
def preprocess_frame(frame):
    # Resize the frame to the size expected by the model
    frame = cv2.resize(frame, (img_size, img_size))
    # Normalize the pixel values (0-1) as done during training
    frame = frame / 255.0
    # Expand dimensions to match the input shape (1, img_size, img_size, 3)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Class labels (update this based on your dataset's classes)
class_labels = ['battery', 'keyboard', 'microwave', 'mobile', 'mouse', 'pcb', 'player', 'printer', 'television', 'washing machine']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    
    # Make prediction
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class with highest probability
    
    # Get the label of the predicted class
    label = class_labels[predicted_class]
    
    # Display the label on the frame
    cv2.putText(frame, f'Detected: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame with the prediction
    cv2.imshow('E-Waste Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

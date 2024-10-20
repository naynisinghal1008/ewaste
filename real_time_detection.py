import numpy as np
import tensorflow as tf
import cv2  # You can also use other libraries to handle image loading
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
from tkinter import Tk

# Load the trained model
model = tf.keras.models.load_model('ewaste_detection_model.keras')

# Class labels (update this based on your dataset's classes)
class_labels = ['battery', 'keyboard', 'microwave', 'mobile', 'mouse', 'pcb', 'player', 'printer', 'television', 'washing machine']

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img_size = 224  # Size should match the model's input size
    img = image.load_img(img_path, target_size=(img_size, img_size))  # Load image with target size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions
def make_prediction(img_path):
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    # Get the index of the highest probability class
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Get the label of the predicted class
    label = class_labels[predicted_class]
    
    return label

# Function to open file dialog and choose an image
def upload_image():
    root = Tk()
    root.withdraw()  # Close the root window
    img_path = filedialog.askopenfilename(title='Select an Image for Prediction')  # Open file dialog to select image
    if img_path:
        # Display the image
        img = cv2.imread(img_path)
        cv2.imshow("Selected Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Predict and display result
        prediction = make_prediction(img_path)
        print(f"Predicted: {prediction}")

# Run the image upload and prediction
upload_image()

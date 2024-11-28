# from flask import Flask, request, jsonify, render_template
# import torch
# from PIL import Image

# app = Flask(__name__)

# # Path to YOLOv5 and best.pt
# YOLOV5_PATH = r'C:\Users\singh\Documents\e-wise\yolov5'  # Replace with your YOLOv5 path
# BEST_MODEL_PATH = r'C:\Users\singh\Documents\e-wise\yolov5\runs\train\exp2\weights\best.pt'  # Replace with your best.pt path

# # Load YOLOv5 model
# model = torch.hub.load(YOLOV5_PATH, 'custom', path=BEST_MODEL_PATH, source='local')

# @app.route('/')
# def home():
#     return render_template('index.html')  # Replace 'index.html' with your webpage file if necessary

# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     try:
#         # Open the uploaded image
#         image = Image.open(file)

#         # Perform detection
#         results = model(image)
#         detections = results.pandas().xyxy[0].to_dict(orient='records')  # Convert to JSON-like structure

#         # Check if any e-waste was detected
#         if len(detections) > 0:
#             return jsonify({
#                 'message': 'E-waste detected!',
#                 'detections': detections
#             })
#         else:
#             return jsonify({'message': 'No e-waste detected.'})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLOv5 model
YOLOV5_PATH = 'C:/Users/singh/Documents/e-wise/yolov5'  # Adjust the path if needed
BEST_MODEL_PATH = 'C:/Users/singh/Documents/e-wise/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load(YOLOV5_PATH, 'custom', path=BEST_MODEL_PATH, source='local')

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part', 'detected': False})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file', 'detected': False})

    # Process the image
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # Run the model on the image
    results = model(image)
    detections = results.pandas().xywh[0]  # Pandas dataframe with results

    # Check if e-waste items are detected
    detected_items = detections[detections['name'].isin(['battery', 'PCB', 'mobile phone'])]

    # If detected items exist, set detected to True
    if not detected_items.empty:
        return jsonify({'message': 'E-waste detected!', 'detected': True})
    else:
        return jsonify({'message': 'No e-waste detected.', 'detected': False})

if __name__ == "__main__":
    app.run(debug=True)

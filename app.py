from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import torch
import tensorflow as tf
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load your trained models (replace with your model paths)
yolo_model = "best.pt"
cnn_model = "tsrd_cnn_model.h5"

def load_models():
    global yolo_model, cnn_model
    try:
        # Load YOLO model (replace 'your_yolo_model.pt' with your actual model path)
        yolo_model = YOLO('best.pt')
        
        # Load CNN model (replace with your actual model path)
        cnn_model = tf.keras.models.load_model('tsrd_cnn_model.h5')
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Class names mapping (customize based on your CNN classes)
# Class names mapping for 58 classes (0-57)
CLASS_NAMES = {
    0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4",
    5: "Class 5", 6: "Class 6", 7: "Class 7", 8: "Class 8", 9: "Class 9",
    10: "Class 10", 11: "Class 11", 12: "Class 12", 13: "Class 13", 14: "Class 14",
    15: "Class 15", 16: "Class 16", 17: "Class 17", 18: "Class 18", 19: "Class 19",
    20: "Class 20", 21: "Class 21", 22: "Class 22", 23: "Class 23", 24: "Class 24",
    25: "Class 25", 26: "Class 26", 27: "Class 27", 28: "Class 28", 29: "Class 29",
    30: "Class 30", 31: "Class 31", 32: "Class 32", 33: "Class 33", 34: "Class 34",
    35: "Class 35", 36: "Class 36", 37: "Class 37", 38: "Class 38", 39: "Class 39",
    40: "Class 40", 41: "Class 41", 42: "Class 42", 43: "Class 43", 44: "Class 44",
    45: "Class 45", 46: "Class 46", 47: "Class 47", 48: "Class 48", 49: "Class 49",
    50: "Class 50", 51: "Class 51", 52: "Class 52", 53: "Class 53", 54: "Class 54",
    55: "Class 55", 56: "Class 56", 57: "Class 57"
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def process_image(image_path):
    """Process image through YOLO detection and CNN classification pipeline"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not read image file'}
            
        original_image = image.copy()

        # Step 1: YOLO Detection
        if yolo_model is None:
            # Simulate YOLO detection for demo purposes
            detections = simulate_yolo_detection(image)
        else:
            results = yolo_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        })
        
        # Step 2: Crop detected regions and classify with CNN
        classified_results = []
        annotated_image = original_image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Draw bounding box on annotated image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop the detected region
            cropped = original_image[y1:y2, x1:x2]
            
            if cropped.size > 0:
                # Classify with CNN
                if cnn_model is None:
                    # Simulate CNN classification for demo purposes
                    predicted_class = np.random.randint(0, 58)
                    confidence = np.random.uniform(0.7, 0.95)
                else:
                    # Preprocess cropped image for CNN
                    processed_crop = preprocess_for_cnn(cropped)
                    prediction = cnn_model.predict(processed_crop)
                    predicted_class = np.argmax(prediction)
                    confidence = float(np.max(prediction))
                
                # Save cropped image
                crop_filename = f'crop_{i}.jpg'
                crop_path = os.path.join(app.config['RESULTS_FOLDER'], crop_filename)
                cv2.imwrite(crop_path, cropped)
                
                # Add label to annotated image
                label = f"{CLASS_NAMES.get(predicted_class, 'Unknown')} ({confidence:.2f})"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                classified_results.append({
                    'class_id': int(predicted_class),
                    'class_name': CLASS_NAMES.get(int(predicted_class), 'Unknown'),
                    'confidence': float(confidence),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'crop_filename': str(crop_filename)
                })

        
        # Save annotated image
        annotated_filename = 'annotated_result.jpg'
        annotated_path = os.path.join(app.config['RESULTS_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        
        return {
            'success': True,
            'detections_count': len(detections),
            'classifications': classified_results,
            'annotated_image': annotated_filename
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def simulate_yolo_detection(image):
    """Simulate YOLO detection for demo purposes"""
    h, w = image.shape[:2]
    # Generate random bounding boxes
    detections = []
    num_detections = np.random.randint(1, 4)
    
    for _ in range(num_detections):
        x1 = np.random.randint(0, w//2)
        y1 = np.random.randint(0, h//2)
        x2 = np.random.randint(x1 + 50, min(x1 + 200, w))
        y2 = np.random.randint(y1 + 50, min(y1 + 200, h))
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': np.random.uniform(0.7, 0.95)
        })
    
    return detections

def preprocess_for_cnn(image):
    """Preprocess cropped image for CNN classification - Updated for 64x64 input size"""
    try:
        # Resize to your CNN input size (64x64)
        target_size = (64, 64)
        resized = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        
        return batch_image
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Return a default preprocessed image if there's an error
        default_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
        return default_image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result = process_image(filepath)
        
        if result['success']:
            return jsonify({
                'success': True,
                'detections_count': result['detections_count'],
                'classifications': result['classifications'],
                'annotated_image': result['annotated_image'],
                'original_filename': filename
            })
        else:
            return jsonify({'success': False, 'error': result['error']})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/results/<filename>')
def get_result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def get_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)

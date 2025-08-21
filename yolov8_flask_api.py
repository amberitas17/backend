from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("yolov8_model.pt")  # replace with your actual model path (e.g., "runs/detect/train/weights/best.pt")

app = Flask(__name__)
CORS(app)  # Allow all origins — useful for Expo Go

# Label map if needed (e.g., class 0 is 'zoetrope')
LABELS = {
    0: 'climate_changed',
    1: 'dialogue_with_time',
    2: 'e3',
    3: 'earth_alive',
    4: 'ecogarden',
    5: 'energy',
    6: 'everyday_science',
    7: 'future_makers',
    8: 'going_viral',
    9: 'kinetic_garden',
    10: 'know_your_poo',
    11: 'laser_maze',
    12: 'phobia2',
    13: 'mirror_maze',
    14: 'savage_garden',
    15: 'singapore_innovations',
    16: 'smart_nation',
    17: 'some_call_it_science',
    18: 'giant_zoetrope',
    19: 'minds_eye',
    20: 'tinkering_studio',
    21: 'urban_mutations',
    22: 'waterworks'
}

@app.route('/detect_base64', methods=['POST'])
def detect_base64():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Extract and decode base64 image
        base64_str = data['image'].split(",")[-1]  # remove "data:image/jpeg;base64," prefix if present
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # TODO: Replace this mock detection with actual YOLO inference
        # For now, return a mock response for testing
        # Uncomment the lines below when you have your YOLO model ready:
        
        results = model(image)[0]
        if len(results.boxes.cls) == 0:
            return jsonify({'exhibit': 'unknown'}), 200
        class_id = int(results.boxes.cls[0])
        exhibit = LABELS.get(class_id, 'unknown')

        return jsonify({'exhibit': exhibit}), 200

    except Exception as e:
        print(f"Error in detect_base64: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Starting Flask API server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

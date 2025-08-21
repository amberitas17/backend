import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import logging
import requests
import time
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for models
roboflow_client = None
emotion_model = None
face_cascade = None

# AssemblyAI configuration
ASSEMBLYAI_API_KEY = "9d46bf92cf684f81b9210bc5574f2580"
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com"

# Roboflow configuration
ROBOFLOW_API_KEY = "s9XwJDaT5rKwSn7ZyM5x"
ROBOFLOW_WORKSPACE = "rich-9cfdj"
ROBOFLOW_AGE_WORKFLOW_ID = "custom-workflow"
ROBOFLOW_EMOTION_WORKFLOW_ID = "detect-and-classify"

# Model configuration - Based on actual training architecture
GENDER_LABELS = ['Male', 'Female']  # gender_dict = {0:"Male", 1:"Female"}

# Emotion labels - Updated to match the provided model
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_models():
    """Initialize Roboflow client, load local emotion model and face detection cascade on startup"""
    global roboflow_client, emotion_model, face_cascade
    
    try:
        # Load Haar cascade for face detection
        logger.info("Loading Haar cascade for face detection...")
        cascade_path = os.path.join('asset', 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detection cascade loaded successfully")
        else:
            logger.error(f"Haar cascade not found at {cascade_path}")
            
        # Load local emotion model
        logger.info("Loading local emotion model...")
        emotion_model_path = os.path.join('asset', 'emotion_model.h5')
        if os.path.exists(emotion_model_path):
            emotion_model = keras.models.load_model(emotion_model_path)
            logger.info("Local emotion model loaded successfully")
        else:
            logger.error(f"Emotion model not found at {emotion_model_path}")
            
        # Initialize Roboflow client for age prediction
        logger.info("Initializing Roboflow client for age prediction...")
        roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        logger.info("Roboflow client initialized successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

def predict_age_with_roboflow(image_base64):
    """Predict age using Roboflow API"""
    try:
        logger.info("Predicting age with Roboflow...")
        
        # Save image temporarily for Roboflow
        temp_image_path = "temp_age_image.jpg"
        
        # Decode base64 image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)
        
        # Run Roboflow age classification workflow
        result = roboflow_client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_AGE_WORKFLOW_ID,
            images={
                "image": temp_image_path
            },
            use_cache=True
        )
        
        # Clean up temp file
        try:
            os.remove(temp_image_path)
        except:
            pass
        
        # Process Roboflow response
        if result and len(result) > 0:
            predictions = result[0].get('predictions', {})
            if 'predictions' in predictions and len(predictions['predictions']) > 0:
                prediction = predictions['predictions'][0]
                class_name = prediction.get('class', '')
                confidence = prediction.get('confidence', 0.0)
                
                logger.info(f"Roboflow age prediction: {class_name} with confidence {confidence}")
                
                # Parse age group based on class
                if "(0-20)" in class_name:
                    age_group = "Child"
                    estimated_age = 15  # Mid-point of 0-20
                elif "(20-40)" in class_name:
                    age_group = "Adult"
                    estimated_age = 30  # Mid-point of 20-40
                elif "40" in class_name or "(40+" in class_name:
                    age_group = "Adult"
                    estimated_age = 50  # Estimated for 40+
                else:
                    age_group = "Adult"
                    estimated_age = 25  # Default
                
                return {
                    'age_group': age_group,
                    'estimated_age': estimated_age,
                    'confidence': confidence,
                    'raw_class': class_name
                }
            
        # Default response if prediction fails
        logger.warning("Roboflow age prediction failed or returned empty results")
        return {
            'age_group': "Adult",
            'estimated_age': 25,
            'confidence': 0.5,
            'raw_class': "unknown"
        }
        
    except Exception as e:
        logger.error(f"Error in Roboflow age prediction: {str(e)}")
        return {
            'age_group': "Adult",
            'estimated_age': 25,
            'confidence': 0.5,
            'raw_class': "error"
        }

# DEPRECATED: Using local emotion model instead
def predict_emotion_with_roboflow(image_base64):
    """DEPRECATED: Predict emotion using Roboflow facial emotion detection API"""
    try:
        logger.info("Predicting emotion with Roboflow...")
        
        # Save image temporarily for Roboflow
        temp_image_path = "temp_emotion_image.jpg"
        
        # Decode base64 image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)
        
        # Run Roboflow emotion detection workflow
        result = roboflow_client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_EMOTION_WORKFLOW_ID,
            images={
                "image": temp_image_path
            },
            use_cache=True
        )
        
        # Clean up temp file
        try:
            os.remove(temp_image_path)
        except:
            pass
        
        # Process Roboflow emotion detection response
        if result and len(result) > 0:
            # Ensure result[0] is a dictionary before calling .get()
            first_result = result[0]
            # Add additional type checking and logging
            logger.info(f"Roboflow emotion response type: {type(first_result)}, content: {first_result}")
            
            if isinstance(first_result, dict):
                predictions_data = first_result.get('predictions', {})
                if 'predictions' in predictions_data and len(predictions_data['predictions']) > 0:
                    # Get all emotion detections
                    detections = predictions_data['predictions']
                    
                    # Find detection with highest confidence
                    best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                    
                    emotion_class = best_detection.get('class', 'neutral')
                    confidence = best_detection.get('confidence', 0.0)
                    
                    # Normalize emotion class name (capitalize first letter)
                    emotion_class = emotion_class.capitalize()
                    
                    # Map to our expected emotion labels if needed
                    emotion_mapping = {
                        'Happy': 'Happy',
                        'Sad': 'Sad', 
                        'Angry': 'Angry',
                        'Fear': 'Fear',
                        'Surprise': 'Surprise',
                        'Disgust': 'Disgust',
                        'Neutral': 'Neutral'
                    }
                    
                    predicted_emotion = emotion_mapping.get(emotion_class, 'Neutral')
                    
                    logger.info(f"Emotion prediction: {predicted_emotion} with confidence {confidence}")
                    
                    # Create emotion probabilities (simulate for compatibility)
                    all_emotions = {}
                    for emotion in EMOTION_LABELS:
                        if emotion == predicted_emotion:
                            all_emotions[emotion] = confidence
                        else:
                            # Distribute remaining probability among other emotions
                            remaining_prob = (1.0 - confidence) / (len(EMOTION_LABELS) - 1)
                            all_emotions[emotion] = remaining_prob
                    
                    return {
                        'predicted_emotion': predicted_emotion,
                        'confidence': confidence,
                        'all_emotions': all_emotions,
                        'detections_count': len(detections)
                    }
            else:
                logger.warning(f"Roboflow result is not a dictionary: {type(first_result)}, value: {first_result}")
                # Handle case where result is a string or other type
                if isinstance(first_result, str):
                    try:
                        # Try to parse as JSON if it's a string
                        import json
                        parsed_result = json.loads(first_result)
                        if isinstance(parsed_result, dict):
                            predictions_data = parsed_result.get('predictions', {})
                            # Continue with normal processing...
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.error(f"Failed to parse string result as JSON: {e}")
        
        # Default response if prediction fails
        logger.warning("Roboflow emotion prediction failed or returned empty results")
        return {
            'predicted_emotion': 'Neutral',
            'confidence': 0.5,
            'all_emotions': {emotion: 0.14 for emotion in EMOTION_LABELS},  # Equal distribution
            'detections_count': 0
        }
        
    except Exception as e:
        logger.error(f"Error in Roboflow emotion prediction: {str(e)}")
        return {
            'predicted_emotion': 'Neutral',
            'confidence': 0.5,
            'all_emotions': {emotion: 0.14 for emotion in EMOTION_LABELS},
            'detections_count': 0
        }

def detect_faces(image):
    """Detect faces in the image using Haar cascade"""
    try:
        if face_cascade is None:
            logger.error("Face cascade not loaded")
            return []
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        logger.info(f"Detected {len(faces)} face(s) in the image")
        return faces
        
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []

def preprocess_image_for_emotion(image, face_coords=None):
    """Preprocess image for emotion model (48x48x1) with optional face cropping"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # If face coordinates are provided, crop to the face region
        if face_coords is not None:
            x, y, w, h = face_coords
            gray = gray[y:y+h, x:x+w]
            
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions (batch_size, height, width, channels)
        processed = np.expand_dims(normalized, axis=[0, -1])
        
        return processed
        
    except Exception as e:
        logger.error(f"Error preprocessing image for emotion: {str(e)}")
        return None

def predict_emotion_local(image_base64):
    """Predict emotion using local .h5 model with face detection"""
    global emotion_model, face_cascade
    
    try:
        if emotion_model is None:
            logger.error("Emotion model not loaded")
            return {
                'predicted_emotion': 'Neutral',
                'confidence': 0.0,
                'all_emotions': {emotion: 0.14 for emotion in EMOTION_LABELS},
                'detections_count': 0,
                'error': 'Emotion model not loaded'
            }
            
        if face_cascade is None:
            logger.error("Face cascade not loaded")
            return {
                'predicted_emotion': 'Neutral',
                'confidence': 0.0,
                'all_emotions': {emotion: 0.14 for emotion in EMOTION_LABELS},
                'detections_count': 0,
                'error': 'Face detection not available'
            }
            
        logger.info("Predicting emotion with local model and face detection...")
        
        # Decode base64 image
        image = decode_base64_image(image_base64)
        if image is None:
            raise ValueError("Failed to decode base64 image")
            
        # Detect faces first
        faces = detect_faces(image)
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return {
                'predicted_emotion': 'Unknown',
                'confidence': 0.0,
                'all_emotions': {emotion: 0.0 for emotion in EMOTION_LABELS},
                'detections_count': 0,
                'error': 'No face detected in the image'
            }
            
        # Use the largest face for prediction
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        logger.info(f"Using largest face at coordinates: ({x}, {y}, {w}, {h})")
        
        # Preprocess for emotion model with face cropping
        processed_image = preprocess_image_for_emotion(image, largest_face)
        if processed_image is None:
            raise ValueError("Failed to preprocess image")
            
        # Make prediction
        predictions = emotion_model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_emotion = EMOTION_LABELS[predicted_class_idx]
        
        # Create emotion probabilities dictionary
        all_emotions = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            all_emotions[emotion] = float(predictions[0][i])
            
        logger.info(f"Local emotion prediction: {predicted_emotion} with confidence {confidence:.3f}")
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'all_emotions': all_emotions,
            'detections_count': len(faces),
            'face_coordinates': largest_face.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error in local emotion prediction: {str(e)}")
        return {
            'predicted_emotion': 'Neutral',
            'confidence': 0.5,
            'all_emotions': {emotion: 1.0/len(EMOTION_LABELS) for emotion in EMOTION_LABELS},
            'detections_count': 0,
            'error': str(e)
        }

def preprocess_image_for_face_analysis(image):
    """Keep face processing functionality for compatibility and face verification"""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Apply basic face enhancement
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply histogram equalization to improve contrast
        enhanced = cv2.equalizeHist(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb
    except Exception as e:
        logger.error(f"Error preprocessing image for face analysis: {str(e)}")
        return image

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    try:
        # Clean the base64 string
        if not base64_string:
            raise ValueError("Empty base64 string provided")
            
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Remove any whitespace and newlines
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '')
        
        # Fix padding if necessary
        # Base64 strings should be divisible by 4
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)
            
        # Validate base64 string contains only valid characters
        import string
        valid_chars = string.ascii_letters + string.digits + '+/='
        if not all(c in valid_chars for c in base64_string):
            raise ValueError("Invalid characters in base64 string")
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Validate decoded data
        if len(image_data) == 0:
            raise ValueError("Decoded image data is empty")
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Validate image
        if pil_image.size[0] == 0 or pil_image.size[1] == 0:
            raise ValueError("Invalid image dimensions")
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        logger.error(f"Base64 string length: {len(base64_string) if 'base64_string' in locals() else 'unknown'}")
        logger.error(f"Base64 string preview: {base64_string[:50] if 'base64_string' in locals() and len(base64_string) > 50 else 'N/A'}...")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'roboflow': roboflow_client is not None,
            'assemblyai': True,  # AssemblyAI is API-based, no local model needed
            'emotion_model': emotion_model is not None,
            'face_cascade': face_cascade is not None
        }
    })

# Removed individual prediction endpoints - only using combined endpoint for actual model predictions

@app.route('/predict/combined', methods=['POST'])
def predict_combined():
    """Predict age using Roboflow and emotion using local .h5 model"""
    try:
        logger.info("Received image analysis request")
        
        # Check if models are loaded
        if roboflow_client is None:
            logger.error("Roboflow client not initialized")
            return jsonify({'error': 'Roboflow client not initialized'}), 500
            
        if emotion_model is None:
            logger.error("Local emotion model not loaded")
            return jsonify({'error': 'Local emotion model not loaded'}), 500
        
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode and preprocess image for face analysis (keep for compatibility)
        image = decode_base64_image(data['image'])
        processed_image = preprocess_image_for_face_analysis(image)
        
        # Predict age using Roboflow
        age_prediction = predict_age_with_roboflow(data['image'])
        
        # Predict emotion using local model
        emotion_prediction = predict_emotion_local(data['image'])
        
        # Check if face detection failed
        if 'error' in emotion_prediction:
            logger.warning(f"Face verification failed: {emotion_prediction['error']}")
            return jsonify({
                'success': False,
                'error': emotion_prediction['error'],
                'message': 'No face detected in the image. Please ensure your face is clearly visible and try again.',
                'predictions': {
                    'age': {
                        'value': 25,
                        'group': 'Unknown',
                        'confidence': 0.0,
                        'roboflow_class': 'no_face'
                    },
                    'gender': {
                        'label': "Unknown",
                        'confidence': 0.0
                    },
                    'emotion': {
                        'label': 'Unknown',
                        'confidence': 0.0
                    },
                    'all_emotions': {emotion: 0.0 for emotion in EMOTION_LABELS},
                    'face_analysis': {
                        'detections_count': 0,
                        'face_coordinates': [],
                        'processed': False
                    }
                }
            }), 200
        
        result = {
            'success': True,
            'predictions': {
                'age': {
                    'value': age_prediction['estimated_age'],
                    'group': age_prediction['age_group'],
                    'confidence': age_prediction['confidence'],
                    'roboflow_class': age_prediction['raw_class']
                },
                'gender': {
                    'label': "Unknown",  # Roboflow workflows don't predict gender yet
                    'confidence': 0.0
                },
                'emotion': {
                    'label': emotion_prediction['predicted_emotion'],
                    'confidence': emotion_prediction['confidence']
                },
                'all_emotions': emotion_prediction['all_emotions'],
                'face_analysis': {
                    'detections_count': emotion_prediction['detections_count'],
                    'face_coordinates': emotion_prediction.get('face_coordinates', []),
                    'processed': True
                }
            }
        }
        
        logger.info(f"Analysis completed: Age={age_prediction['estimated_age']} ({age_prediction['age_group']}), Emotion={emotion_prediction['predicted_emotion']} (local model)")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in combined prediction: {str(e)}")
        logger.error(f"Full error traceback:\n{error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': 'Check Flask server logs for full error details'
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using AssemblyAI API"""
    try:
        logger.info("Received audio transcription request")
        
        # Check if audio file is in request
        if 'audio' not in request.files:
            logger.error("No audio file provided")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save audio file temporarily
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join('temp', filename)
        
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        audio_file.save(temp_path)
        logger.info(f"Audio file saved to: {temp_path}")
        
        # Upload audio file to AssemblyAI
        logger.info("Uploading audio to AssemblyAI...")
        
        headers = {
            "authorization": ASSEMBLYAI_API_KEY
        }
        
        # Upload the audio file
        with open(temp_path, "rb") as f:
            upload_response = requests.post(
                ASSEMBLYAI_BASE_URL + "/v2/upload",
                headers=headers,
                data=f
            )
        
        if upload_response.status_code != 200:
            logger.error(f"Failed to upload audio: {upload_response.text}")
            return jsonify({'error': 'Failed to upload audio to AssemblyAI'}), 500
        
        audio_url = upload_response.json()["upload_url"]
        logger.info(f"Audio uploaded successfully, URL: {audio_url}")
        
        # Submit transcription request
        data = {
            "audio_url": audio_url,
            "speech_model": "universal"
        }
        
        transcription_response = requests.post(
            ASSEMBLYAI_BASE_URL + "/v2/transcript",
            json=data,
            headers=headers
        )
        
        if transcription_response.status_code != 200:
            logger.error(f"Failed to submit transcription: {transcription_response.text}")
            return jsonify({'error': 'Failed to submit transcription to AssemblyAI'}), 500
        
        transcript_id = transcription_response.json()['id']
        logger.info(f"Transcription submitted, ID: {transcript_id}")
        
        # Poll for completion
        polling_endpoint = ASSEMBLYAI_BASE_URL + "/v2/transcript/" + transcript_id
        
        max_attempts = 60  # Maximum 3 minutes of polling (60 * 3 seconds)
        attempts = 0
        
        while attempts < max_attempts:
            transcription_result = requests.get(polling_endpoint, headers=headers).json()
            
            if transcription_result['status'] == 'completed':
                transcript_text = transcription_result['text']
                logger.info(f"AssemblyAI transcription completed: '{transcript_text}'")
                
                # Generate response based on detected keywords
                response_text = generate_keyword_response(transcript_text)
                
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
                
                return jsonify({
                    'success': True,
                    'transcript': transcript_text,  # Original transcribed text
                    'keyword_response': response_text,  # Keyword-based response for AI
                    'original_transcript': transcript_text  # Keep for backward compatibility
                })
                
            elif transcription_result['status'] == 'error':
                error_msg = transcription_result.get('error', 'Unknown error')
                logger.error(f"AssemblyAI transcription failed: {error_msg}")
                return jsonify({'error': f'Transcription failed: {error_msg}'}), 500
            
            else:
                logger.info(f"Transcription in progress... Status: {transcription_result['status']}")
                time.sleep(3)
                attempts += 1
        
        # If we reach here, polling timed out
        logger.error("Transcription polling timed out")
        return jsonify({'error': 'Transcription timed out'}), 500
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in audio transcription: {str(e)}")
        logger.error(f"Full error traceback:\n{error_details}")
        
        # Clean up temp file if it exists
        try:
            if 'temp_path' in locals():
                os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': str(e),
            'details': 'Check Flask server logs for full error details'
        }), 500

def generate_keyword_response(transcript):
    """Generate response based on detected keywords in transcript"""
    
    # Convert to lowercase for keyword matching
    text = transcript.lower()
    
    # Define keyword patterns and responses
    keyword_responses = {
        'hello': "Hello",
        'hi': "Hello", 
        'good morning': "Good morning",
        'good afternoon': "Good afternoon",
        'opening hours': "What are the opening hours?",
        'hours': "What are the opening hours?",
        'food': "Where is the food court?",
        'eat': "Where is the food court?",
        'restaurant': "Where is the food court?",
        'kidsstop': "How do I get to KidsSTOP?",
        'children': "What activities are suitable for young children?",
        'kids': "What activities are suitable for young children?",
        'exhibition': "What exhibitions are currently showing?",
        'show': "What exhibitions are currently showing?",
        'park': "Where can I park?",
        'parking': "Where can I park?",
        'program': "Are there any special programs today?",
        'workshop': "Are there any educational workshops today?",
        'interactive': "Where can I find interactive science experiments?",
        'science': "Tell me about the interactive exhibits",
        'help': "How can I help you today?",
        'thank': "You're welcome!"
    }
    
    # Check for keyword matches
    for keyword, response in keyword_responses.items():
        if keyword in text:
            logger.info(f"Detected keyword '{keyword}' in transcript: '{transcript}'")
            return response
    
    # If no keywords detected, return random response
    random_responses = [
        "What are the opening hours?",
        "Where is the food court?", 
        "How do I get to KidsSTOP?",
        "What exhibitions are showing?",
        "Are there any special programs?",
        "Where can I park?",
        "What activities are for children?",
        "Tell me about the interactive exhibits"
    ]
    
    import random
    random_response = random.choice(random_responses)
    logger.info(f"No keywords detected in '{transcript}', using random response: '{random_response}'")
    return random_response


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

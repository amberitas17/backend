# Face Verification Flask Backend

A Python Flask backend for Expo React Native applications that provides face verification services using trained Keras models for age-gender detection and emotion recognition.

## Features

- **Age & Gender Detection**: Predicts age range and gender from face images
- **Emotion Recognition**: Identifies emotions from facial expressions
- **Combined Analysis**: Performs both age-gender and emotion analysis in a single request
- **RESTful API**: Easy integration with React Native frontend
- **Base64 Image Support**: Accepts images as base64 encoded strings
- **CORS Enabled**: Ready for cross-origin requests from mobile apps

## Requirements

- Python 3.9
- Virtual environment (recommended)
- Trained Keras models (.h5 files)

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are in place:**
   - Place your trained models in the `asset/` directory:
     - `age_gender_model.h5`
     - `emotion_model.h5`

## Usage

### Starting the Server

```bash
python run.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```
GET /health
```
Returns server status and model loading status.

#### 2. Age & Gender Prediction
```
POST /predict/age-gender
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "age": {
      "range": "25-32",
      "confidence": 0.85
    },
    "gender": {
      "label": "Female",
      "confidence": 0.92
    }
  }
}
```

#### 3. Emotion Prediction
```
POST /predict/emotion
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "emotion": {
      "label": "Happy",
      "confidence": 0.78
    },
    "all_emotions": {
      "Angry": 0.02,
      "Disgust": 0.01,
      "Fear": 0.03,
      "Happy": 0.78,
      "Sad": 0.05,
      "Surprise": 0.08,
      "Neutral": 0.03
    }
  }
}
```

#### 4. Combined Prediction
```
POST /predict/combined
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "age": {
      "range": "25-32",
      "confidence": 0.85
    },
    "gender": {
      "label": "Female",
      "confidence": 0.92
    },
    "emotion": {
      "label": "Happy",
      "confidence": 0.78
    },
    "all_emotions": {
      "Angry": 0.02,
      "Disgust": 0.01,
      "Fear": 0.03,
      "Happy": 0.78,
      "Sad": 0.05,
      "Surprise": 0.08,
      "Neutral": 0.03
    }
  }
}
```

## React Native Integration

### Example Usage in React Native

```javascript
const analyzeImage = async (imageBase64) => {
  try {
    const response = await fetch('http://your-server-ip:5000/predict/combined', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageBase64
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('Age:', result.predictions.age.range);
      console.log('Gender:', result.predictions.gender.label);
      console.log('Emotion:', result.predictions.emotion.label);
    }
  } catch (error) {
    console.error('Error analyzing image:', error);
  }
};
```

### Converting Image to Base64 in React Native

```javascript
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

const convertImageToBase64 = async (imageUri) => {
  try {
    const manipResult = await manipulateAsync(
      imageUri,
      [{ resize: { width: 224, height: 224 } }],
      { compress: 0.8, format: SaveFormat.JPEG, base64: true }
    );
    
    return manipResult.base64;
  } catch (error) {
    console.error('Error converting image:', error);
    return null;
  }
};
```

## Configuration

### Model Labels
Update the labels in `config.py` to match your trained models:

```python
# Adjust these based on your model outputs
AGE_RANGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
GENDER_LABELS = ['Male', 'Female']
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
```

### Input Sizes
Adjust image input sizes based on your model requirements:

```python
AGE_GENDER_INPUT_SIZE = (224, 224)  # Adjust based on your model
EMOTION_INPUT_SIZE = (48, 48)       # Adjust based on your model
```

## File Structure

```
backend/
├── app.py              # Main Flask application
├── run.py              # Startup script
├── config.py           # Configuration settings
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── asset/             # Model files directory
    ├── age_gender_model.h5
    └── emotion_model.h5
```

## Error Handling

The API includes comprehensive error handling:
- Model loading errors
- Image decoding errors
- Prediction errors
- Input validation errors

All errors return appropriate HTTP status codes and error messages.

## Logging

The application logs important events and errors to both console and `app.log` file for debugging purposes.

## Production Deployment

For production deployment:
1. Set `DEBUG = False` in `config.py`
2. Configure appropriate CORS origins
3. Use a production WSGI server like Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure your .h5 files are compatible with the TensorFlow version
2. **Memory Issues**: Consider reducing batch size or image resolution for resource-constrained environments
3. **CORS Issues**: Update CORS_ORIGINS in config.py for your specific frontend domain

### Dependencies Compatibility

This backend is designed for Python 3.9 with compatible library versions:
- TensorFlow 2.13.0
- Keras 2.13.1
- Flask 2.3.3

## Support

For issues or questions, please check the logs in `app.log` for detailed error information.

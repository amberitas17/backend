import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model("asset/emotion_model.h5")

# Save as TensorFlow SavedModel (no save_format arg, just directory path)
model.export("asset/emotion_model_saved")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("asset/emotion_model_saved")
tflite_model = converter.convert()

# Write out the .tflite file
with open("asset/emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: emotion_model.tflite")

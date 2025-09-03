import tensorflow as tf
import numpy as np
import cv2
import os
import json

# === Model Setup ===
MODEL_PATH = "model.tflite"
LABEL_MAPPING_FILE = "label_mapping.json"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Utility Functions ===

def preprocess_image(image_path, mode="0_1"):
    """Preprocess image for TFLite inference."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    img = cv2.resize(img, (w, h))

    img = img.astype(np.float32) / 255.0  # [0,1] normalization
    return np.expand_dims(img, axis=0)

def classify_with_labels(image_path, labels):
    """Classify image with a given label order."""
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx = int(np.argmax(output_data))
    return {
        "label": labels[top_idx],
        "confidence": float(output_data[top_idx]),
        "all_scores": {labels[i]: float(output_data[i]) for i in range(len(labels))}
    }

def save_label_order(label_order, path=LABEL_MAPPING_FILE):
    with open(path, "w") as f:
        json.dump({"labels": label_order}, f)
    print(f"✅ Saved label order: {label_order} -> {path}")

def load_label_order(path=LABEL_MAPPING_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data["labels"]
    return None

def remove_test_image(image_path):
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"🗑️ Removed test image: {image_path}")

def auto_detect_label_order(image_path, true_label):
    """
    Compare both possible label orders against a known test image's true label.
    Saves the correct order and removes the test image.
    """
    label_orders = [
        ["dialogue_with_time", "earth_alive"],  # option A
        ["earth_alive", "dialogue_with_time"]   # option B
    ]

    results = []
    for order in label_orders:
        prediction = classify_with_labels(image_path, order)
        results.append((order, prediction))

    # Check which order matches the true label
    for order, prediction in results:
        print(f"\nTesting order {order}: {prediction}")
        if prediction["label"] == true_label:
            print(f"✅ Correct order detected: {order}")
            save_label_order(order)
            remove_test_image(image_path)
            return order

    # If no exact match, choose the one with higher confidence for true_label
    best_order = None
    best_conf = -1
    for order, prediction in results:
        conf = prediction["all_scores"].get(true_label, 0)
        if conf > best_conf:
            best_conf = conf
            best_order = order
    print(f"\n⚠️ No exact match. Using closest order: {best_order}")
    save_label_order(best_order)
    remove_test_image(image_path)
    return best_order

# === Main Classification Function ===

def classify_image(image_path):
    labels = load_label_order()
    if not labels:
        raise RuntimeError("❌ Label order not set. Run auto-detect first with a test image.")

    prediction = classify_with_labels(image_path, labels)
    return prediction

# === Example Usage ===
if __name__ == "__main__":
    # STEP 1: Run this ONCE with a known test image + true label
    # auto_detect_label_order("test_images/earth_alive2.jpg", "earth_alive")

    # STEP 2: After mapping is saved, use classify_image() normally
    result = classify_image("test_images/compressed-tinyjpg-1.jpg")
    print("\nFinal Prediction:", result)

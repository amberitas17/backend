from PIL import Image
import glob
import os

# Import your classify_image function from your main code
from azure import classify_image  # replace with your actual filename

# Paths to your test images
test_images = {
    "dialogue_with_time": glob.glob("test_images/1.jpg"),
    "earth_alive": glob.glob("test_images/compressed-tinyjpg-1.jpg")
}

for label, paths in test_images.items():
    print(f"\nTesting images for: {label}")
    for img_path in paths:
        img = Image.open(img_path)
        result = classify_image(img)
        print(f"{os.path.basename(img_path)} -> {result}")
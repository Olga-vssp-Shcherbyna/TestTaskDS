import numpy as np
import tensorflow as tf
from PIL import Image
import os
import subprocess


def train_cv_model_if_needed():
    cv_model_path = 'best_model.keras'
    if not os.path.exists(cv_model_path):
        print("CV model not found. Starting training...")
        subprocess.run(['python', 'cv_model_training.py'])
    return cv_model_path


# CV model loading
def load_cv_model():
    cv_model_path = train_cv_model_if_needed()
    return tf.keras.models.load_model(cv_model_path)


model = load_cv_model()

# Dictionary for converting labels into animal names
label_map = {
    0: 'cat',
    1: 'dog',
    2: 'cow',
    3: 'sheep',
    4: 'horse',
    5: 'elephant',
    6: 'spider',
    7: 'butterfly',
    8: 'chicken',
    9: 'squirrel'
}


def decode_predictions(predictions):
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class with the highest probability
    return label_map.get(predicted_class, "Unknown")  # Return the name of the animal or "Unknown"


def classify_animal(image_file):
    try:
        # Open the image with PIL
        image = Image.open(image_file)
        image = image.convert("RGB")  # Convert to RGB

        image = tf.image.resize(image, (128, 128))  # Resize
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normalize
        img_array = np.array(image)  # Convert to an array
        img_array = img_array / 255.0  # Normalizing pixel values

        img_array = np.expand_dims(img_array, axis=0)  # Adding a dimension for batch processing

        # Pass img_array to the model
        predictions = model.predict(img_array)
        return decode_predictions(predictions)  # Returning results
    except Exception as e:
        print(f"Error in classify_animal: {e}")
        return None

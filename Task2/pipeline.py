import os
import subprocess
import spacy

from animal_classification import classify_animal
from user_text_input_processing import process_text

global ner_model_path


def train_ner_model_if_needed():
    ner_model_path = 'trained_model'
    if not os.path.exists(ner_model_path):
        print("NER model not found. Starting training...")
        subprocess.run(['python', 'ner_model_training.py'])


def processing_pipeline():
    train_ner_model_if_needed()
    image_dir = input("Enter path to your image including image name or 'exit' to quit: ")
    if image_dir.lower() == 'exit':
        print("Exiting the program.")
        return
    text_data = input("Enter image description: ")
    print("Processing...")

    nlp = spacy.load(ner_model_path)
    ner = nlp.get_pipe("ner")
    labels = ner.labels

    # Text processing
    detected_animal = process_text(text_data, labels)

    # Images processing
    animal_from_image = classify_animal(image_dir)

    # Data comparing
    if detected_animal and animal_from_image:
        is_correct = detected_animal[0].lower() == animal_from_image.lower()
    else:
        is_correct = False  # If the animal is not identified from image or text

    print(is_correct)

    processing_pipeline()


if __name__ == '__main__':
    processing_pipeline()

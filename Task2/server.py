import os

from flask import Flask, render_template, request
import spacy

from animal_classification import classify_animal
from user_text_input_processing import process_text
from pipeline import train_ner_model_if_needed

app = Flask(__name__)  # Flask initialization


@app.route("/")
def home():
    return render_template('index.html')  # Display start page


@app.route("/process", methods=["POST"])
def process():
    train_ner_model_if_needed()
    if "file" not in request.files:  # Check if file is in the request
        return "No file part", 400  # If not - 404 error

    file = request.files["file"]
    if file.filename == "":  # Check if filename exists
        return "No selected file", 400  # If not - 404 error

    static_path = 'static/uploads'
    if not os.path.exists(static_path):
        os.makedirs(static_path)

    # File storage
    image_url = f"static/uploads/{file.filename}"
    file.save(image_url)

    text_data = request.form['description']  # Get the text from web form

    nlp = spacy.load('trained_model')
    ner = nlp.get_pipe("ner")
    labels = ner.labels

    # Text processing
    detected_animal = process_text(text_data.lower(), labels)

    # Image processing
    animal_from_image = classify_animal(image_url)

    # Data comparing
    if detected_animal and animal_from_image:
        is_correct = detected_animal[0].lower() == animal_from_image.lower()
    else:
        is_correct = False  # If the animal is not identified from image or text

    return render_template("result.html", image=image_url, text=text_data, result=is_correct)


if __name__ == '__main__':
    app.run(debug=False)

import spacy


def process_text(text, labels):
    nlp = spacy.load('trained_model')  # Loading our trained NER model
    doc = nlp(text)
    detected_animals = [ent.text for ent in doc.ents if ent.label_ in labels]
    return detected_animals

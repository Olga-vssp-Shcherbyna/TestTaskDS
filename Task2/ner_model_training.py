import spacy
from spacy.training import Example
import random

from text_dataset_creation import train_data

# Pre-trained nlp model loading
nlp = spacy.load("en_core_web_lg")

# Create NER component
ner = nlp.get_pipe("ner")

# Add labels to the NER
for item in train_data:
    for label in item['entities']:
        ner.add_label(label[2])

# Optimizer initialization
optimizer = nlp.create_optimizer()

# Learning parameters
n_iter_def = 30  # Epochs number
batch_size_def = 8  # Batch size
learning_rate_def = 0.0001  # Learning rate
dropout_rate_def = 0.1  # Dropout

n_iter_input = input(f"Enter number of epochs (default is 30): ")
batch_size_input = input(f"Enter batch size (default is 8): ")
learning_rate_input = input(f"Enter learning rate (default is 0.0001): ")
dropout_rate_input = input(f"Enter dropout rate (default is 0.1): ")

n_iter = int(n_iter_input) if n_iter_input else n_iter_def
batch_size = int(batch_size_input) if batch_size_input else batch_size_def
learning_rate = float(learning_rate_input) if learning_rate_input else learning_rate_def
dropout_rate = float(dropout_rate_input) if dropout_rate_input else dropout_rate_def

# Learning cycle
for epoch in range(n_iter):
    print(f"Epoch {epoch + 1}/{n_iter}")
    # Mix training data
    random.shuffle(train_data)

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        examples = []

        for item in batch:
            text = item['text']
            entities = [(ent[0], ent[1], ent[2]) for ent in item['entities']]

            # Create Doc object
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": entities})
            examples.append(example)

        # Update model
        nlp.update(examples, drop=dropout_rate, losses={})

# Model storage
nlp.to_disk('trained_model')

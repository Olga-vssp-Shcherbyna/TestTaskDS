import spacy

from text_dataset_creation import test_data

# Load our trained NER model
nlp = spacy.load('trained_model')

# Calculation variables
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0

# Evaluate model on test data
for item in test_data:
    text = item['text']
    true_entities = [(ent[0], ent[1], ent[2]) for ent in item['entities']]

    # Predict animals
    doc = nlp(text)
    predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    # Converting lists into sets for easier counting
    true_set = set(true_entities)
    pred_set = set(predicted_entities)

    # Calculate true positives, false positives, and false negatives
    true_positives = true_set.intersection(pred_set)
    false_positives = pred_set - true_set
    false_negatives = true_set - pred_set

    total_true_positives += len(true_positives)
    total_false_positives += len(false_positives)
    total_false_negatives += len(false_negatives)

# Calculating accuracy, recall, and F1
precision = total_true_positives / (total_true_positives + total_false_positives) if (
                                                                                             total_true_positives + total_false_positives) > 0 else 0
recall = total_true_positives / (total_true_positives + total_false_negatives) if (
                                                                                          total_true_positives + total_false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

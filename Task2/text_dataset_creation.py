from sklearn.model_selection import train_test_split
import json

# Read text data from file

with open('Animals_in_text_dataset.txt', 'r') as file:
    try:
        data = json.load(file)  # File parsing
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")

# Split into training (60%), validation (20%), and test (20%) data
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)  # 60% training, 40% remaining
val_data, test_data = train_test_split(temp_data, test_size=0.5,
                                       random_state=42)  # 50% for validation and 50% for testing

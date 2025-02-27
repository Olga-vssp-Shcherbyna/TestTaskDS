from cv_model_training import model
from images_dataset_creation import test_dataset

test_loss, test_accuracy = model.evaluate(test_dataset) # Model evaluation on test part of our dataset

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

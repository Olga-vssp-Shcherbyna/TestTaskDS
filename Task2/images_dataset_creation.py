import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def train_preprocess(image, label):
    image = tf.image.resize(image, (128, 128))  # Resize
    image = tf.image.random_flip_left_right(image)  # RandomHorizontalFlip(0.5)
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normalize
    return image, label


def val_preprocess(image, label):
    image = tf.image.resize(image, (128, 128))  # Resize
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normalize
    return image, label


def load_and_preprocess_image(img_path, img_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalization
    return img


def create_dataset(data_dir, img_size=(128, 128)):
    images = []
    labels = []

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            print(f"Processing class: {label}")  # Class name output
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                images.append(load_and_preprocess_image(img_path, img_size))
                labels.append(label)  # Add class label to the label list

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return tf.data.Dataset.from_tensor_slices((images, labels_encoded))


# Upload data
dataset = create_dataset('Animal-10-split')

dataset = dataset.shuffle(buffer_size=10000)

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = int(0.1 * len(dataset))  # 10% for validation
test_size = len(dataset) - train_size - val_size  # 10% for testing

train_dataset = dataset.take(train_size)  # Training dataset
validation_dataset = dataset.skip(train_size).take(val_size)  # Validation dataset
test_dataset = dataset.skip(train_size + val_size)  # Test dataset

# Pack data into batches
batch_size = 64
train_dataset = train_dataset.map(train_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(val_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(val_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

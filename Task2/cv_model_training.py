from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from images_dataset_creation import train_dataset, validation_dataset

# Create callback for best model storage
checkpoint = ModelCheckpoint(
    'best_model.keras',  # Path to saved model
    monitor='val_accuracy',  # Metrics to monitor
    save_best_only=True,  # Keep only the best model
    mode='max',  # Metric maximization
    verbose=1  # Display a save message
)

# Create callback for early stopping if model is not improving in 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metrics to monitor
    patience=5,  # Number of epochs after which to stop learning if there has been no improvement
    mode='min',  # Metric minimization
    verbose=1  # Display stop message
)

# Load base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze model layers

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # Monitor validation loss
    factor=0.5,  # Reduce learning rate by 2 times (can be changed)
    patience=3,  # Wait for 3 epochs without improvement before reducing
    min_lr=1e-6,  # Minimum learning rate to not stop learning
    verbose=1  # Print messages to the console
)

unfreezed_layers_def = 40
dense_inner_layers_def = 512
dropout_def = 0.3
l2_value_def = 0.01
epochs_def = 10
learning_rate_def=0.005
momentum_def=0.9
weight_decay_def=0.0005

dense_inner_layers_input = input("Enter number of inner layers (default is 512): ")
dropout_input = input("Enter dropout value (default is 0.3): ")
l2_value_input = input("Enter l2 value (default is 0.01): ")
unfreezed_layers_input = input("Enter unfreezed layers number (default is 40): ")
epochs_input = input("Enter number of epochs (default is 10): ")
learning_rate_input = input("Enter learing rate (default is 0.005): ")
momentum_input = input("Enter momentum (default is 0.9): ")
weight_decay_input = input("Enter weight decay (default is 0.0005): ")

dense_inner_layers = int(dense_inner_layers_input) if dense_inner_layers_input else dense_inner_layers_def
dropout = float(dropout_input) if dropout_input else dropout_def
l2_value = float(l2_value_input) if l2_value_input else l2_value_def
unfreezed_layers = int(unfreezed_layers_input) if unfreezed_layers_input else unfreezed_layers_def
epochs = int(epochs_input) if epochs_input else epochs_def
learning_rate = float(learning_rate_input) if learning_rate_input else learning_rate_def
momentum = float(momentum_input) if momentum_input else momentum_def
weight_decay = float(weight_decay_input) if weight_decay_input else weight_decay_def

# Unfreeze top n layers
for layer in base_model.layers[-unfreezed_layers:]:
    layer.trainable = True

# Compose model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(dense_inner_layers, activation='relu', kernel_regularizer=l2(l2_value)),
    Dropout(dropout),
    Dense(10, activation='softmax')  # number of classes = 10 for Animal-10
])

# Compile model
model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

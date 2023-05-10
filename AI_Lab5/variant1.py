# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

# Load the dataset
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train_full = X_train_full.reshape((60000, 784)).astype('float32') / 255.0
X_test = X_test.reshape((10000, 784)).astype('float32') / 255.0
y_train_full = keras.utils.to_categorical(y_train_full)
y_test = keras.utils.to_categorical(y_test)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

# Define the MLP architecture
model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# Define hyperparameters
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [1, 32, 128]
num_hidden_layers = [0, 1, 2]
hidden_layer_widths = [64, 128, 256]
loss_functions = ['mse', 'mae', 'categorical_crossentropy']

# Train and evaluate the model for different hyperparameters
results = []
for lr in learning_rates:
    for batch_size in batch_sizes:
        for num_hidden in num_hidden_layers:
            for width in hidden_layer_widths:
                for loss_fn in loss_functions:
                    print("Training model with lr={}, batch_size={}, num_hidden={}, width={}, loss_fn={}".format(
                        lr, batch_size, num_hidden, width, loss_fn))
                    model = keras.models.Sequential()
                    model.add(keras.layers.Dense(
                        width, activation='relu', input_shape=(784,)))
                    for i in range(num_hidden):
                        model.add(keras.layers.Dense(width, activation='relu'))
                    model.add(keras.layers.Dense(10, activation='softmax'))
                    model.compile(loss=loss_fn, optimizer=keras.optimizers.SGD(
                        lr=lr), metrics=['accuracy'])
                    history = model.fit(
                        X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_valid, y_valid))
                    val_accuracy = history.history['val_accuracy'][-1]
                    results.append({'lr': lr, 'batch_size': batch_size, 'num_hidden': num_hidden,
                                    'width': width, 'loss_fn': loss_fn, 'history': history,
                                    'val_accuracy': val_accuracy})
# Print results
for result in results:
    print(result)

# Plot loss and accuracy curves

for result in results:
    plt.plot(result['history'].history['loss'], label='Training Loss')
    plt.plot(result['history'].history['val_loss'], label='Validation Loss')
    plt.title('Loss - LR={}, Batch Size={}, Num Hidden Layers={}, Width={}, Loss Function={}'.format(
        result['lr'], result['batch_size'], result['num_hidden'], result['width'], result['loss_fn']))
    plt.legend()
    plt.show()

plt.plot(result['history'].history['accuracy'], label='Training Accuracy')
plt.plot(result['history'].history['val_accuracy'],
         label='Validation Accuracy')
plt.title('Accuracy - LR={}, Batch Size={}, Num Hidden Layers={}, Width={}, Loss Function={}'.format(
    result['lr'], result['batch_size'], result['num_hidden'], result['width'], result['loss_fn']))
plt.legend()
plt.show()

# Print best results
best_result = max(results, key=lambda x: x['val_accuracy'])
print("Best result:\n", best_result)
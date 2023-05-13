# Import necessary libraries
import itertools

import matplotlib.pyplot as plt
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
def create_model(width, num_hidden, loss_fn, lr):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(width, activation='relu', input_shape=(784,)))
    for i in range(num_hidden):
        model.add(keras.layers.Dense(width, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss=loss_fn, optimizer=keras.optimizers.SGD(lr=lr), metrics=['accuracy'])
    return model


# Define hyperparameters
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [1, 32, 128]
num_hidden_layers = [0, 1, 2]
hidden_layer_widths = [64, 128, 256]
loss_functions = ['mse', 'mae', 'categorical_crossentropy']

# Create all possible combinations of hyperparameters
hyperparameters = itertools.product(learning_rates, batch_sizes, num_hidden_layers, hidden_layer_widths, loss_functions)

# Train and evaluate the model for different hyperparameters
results = []
for lr, batch_size, num_hidden, width, loss_fn in hyperparameters:
    print("Training model with lr={}, batch_size={}, num_hidden={}, width={}, loss_fn={}".format(
        lr, batch_size, num_hidden, width, loss_fn))
    model = create_model(width, num_hidden, loss_fn, lr)

    # Add the early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model with the early stopping callback
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_valid, y_valid),
        callbacks=[early_stopping])

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

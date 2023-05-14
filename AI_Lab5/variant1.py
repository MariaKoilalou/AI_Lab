# Import necessary libraries
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
batch_sizes = [32, 64, 128]
num_hidden_layers = [0, 1, 2]
hidden_layer_widths = [64, 128, 256]
loss_functions = ['mse', 'mae', 'categorical_crossentropy']

# Train and evaluate the model for different hyperparameters
results = []
for batch_size in batch_sizes:
    for lr in learning_rates:
        print(
            "Training model with lr={}, batch_size={}, num_hidden_layers = 1, hidden_layer_widths = 128 , "
            "loss_functions = categorical_crossentropy".format(
                lr, batch_size))
        model = create_model(width=128, num_hidden=1, loss_fn='categorical_crossentropy', lr=lr)

        # Add the early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True)

        # Fit the model with the early stopping callback
        history = model.fit(
            X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_valid, y_valid),
            callbacks=[early_stopping])

        val_accuracy = history.history['val_accuracy'][-1]
        results.append({'lr': lr, 'batch_size': batch_size, 'num_hidden': 1,
                        'width': 128, 'loss_fn': 'categorical_crossentropy', 'history': history,
                        'val_accuracy': val_accuracy})

for layers in num_hidden_layers:
    print("Training model with lr=0.01, batch_size=64, num_hidden_layers = {}, hidden_layer_widths = 128 , "
          "loss_functions = categorical_crossentropy".format(layers))
    model = create_model(width=128, num_hidden=layers, loss_fn='categorical_crossentropy', lr=0.01)
    # Add the early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model with the early stopping callback
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping])

    val_accuracy = history.history['val_accuracy'][-1]
    results.append(
        {'lr': 0.01, 'batch_size': 64, 'num_hidden': layers, 'width': 128, 'loss_fn': 'categorical_crossentropy',
         'history': history,
         'val_accuracy': val_accuracy})

for width in hidden_layer_widths:
    print("Training model with lr=0.01, batch_size=64, num_hidden_layers = 1, hidden_layer_width = {} , "
          "loss_functions = categorical_crossentropy".format(width))
    model = create_model(width=width, num_hidden=1, loss_fn='categorical_crossentropy', lr=0.01)
    # Add the early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model with the early stopping callback
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping])

    val_accuracy = history.history['val_accuracy'][-1]
    results.append(
        {'lr': 0.01, 'batch_size': 64, 'num_hidden': 1, 'width': width, 'loss_fn': 'categorical_crossentropy',
         'history': history,
         'val_accuracy': val_accuracy})

for loss_fn in loss_functions:
    print("Training model with lr=0.01, batch_size=64, num_hidden_layers = 1, hidden_layer_widths = 128 , "
          "loss_functions = {}".format(loss_fn))
    model = create_model(width=128, num_hidden=1, loss_fn=loss_fn, lr=0.01)
    # Add the early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Fit the model with the early stopping callback
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping])

    val_accuracy = history.history['val_accuracy'][-1]
    results.append(
        {'lr': 0.01, 'batch_size': 64, 'num_hidden': 1, 'width': 128, 'loss_fn': loss_fn, 'history': history,
         'val_accuracy': val_accuracy})

# Print results
for result in results:
    print(result)

plt.subplots_adjust(hspace=1, wspace=2)

for result in results:
    print(result)
    plt.plot(result['history'].history['loss'], label='Training Loss')
    plt.plot(result['history'].history['val_loss'], label='Validation Loss')
    plt.title('Loss - LR={}, Batch Size={}, Loss Function={}'.format(
        result['lr'], result['batch_size'], result['loss_fn']))
    plt.xlabel('Learning Steps')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

    plt.plot(result['history'].history['accuracy'], label='Training Accuracy')
    plt.plot(result['history'].history['val_accuracy'],label='Validation Accuracy')
    plt.title('Accuracy - LR={}, Batch Size={}, Num Hidden Layers={}, Width={}, Loss Function={}'.format(
        result['lr'], result['batch_size'], result['num_hidden'], result['width'], result['loss_fn']))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Print best results
best_result = max(results, key=lambda x: x['val_accuracy'])
print("Best result:\n", best_result)

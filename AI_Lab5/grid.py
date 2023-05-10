# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV
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
def create_model(num_hidden_layers=1, hidden_layer_widths=128, loss_function='categorical_crossentropy',
                 learning_rate=0.01):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_layer_widths, activation='relu', input_shape=(784,)))
    for i in range(num_hidden_layers):
        model.add(keras.layers.Dense(hidden_layer_widths, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    return model


# Define hyperparameters and their possible values
param_grid = {
    'num_hidden_layers': [1, 2],
    'hidden_layer_widths': [64, 128, 256],
    'loss_function': ['mse', 'mae', 'categorical_crossentropy'],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Create Grid Search object and fit the data
grid_search = GridSearchCV(keras.wrappers.scikit_learn.KerasClassifier(create_model),
                           param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best hyperparameters and their accuracy
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_.model

# Evaluate the model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)

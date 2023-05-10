import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, SparseCategoricalCrossentropy

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0


# Define neural network architecture
def create_model(num_hidden_layers, width, loss_function):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # Flatten input images
    for _ in range(num_hidden_layers):
        model.add(Dense(width, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model


# Define hyperparameters
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [1, 32, 128]
num_hidden_layers = [0, 1, 2]
widths = [10, 50, 100]
loss_functions = [MeanSquaredError(), MeanAbsoluteError(), SparseCategoricalCrossentropy(from_logits=True)]

# Train and evaluate models for all hyperparameter combinations
for lr in learning_rates:
    for batch_size in batch_sizes:
        for num_layers in num_hidden_layers:
            for width in widths:
                for loss_func in loss_functions:
                    model = create_model(num_layers, width, loss_func)
                    model.fit(train_images, train_labels, batch_size=batch_size, epochs=10, verbose=0)
                    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
                    print(
                        f"lr={lr:.4f}, batch_size={batch_size}, num_layers={num_layers}, width={width}, loss_function={type(loss_func).__name__}: test_acc={test_acc:.4f}")

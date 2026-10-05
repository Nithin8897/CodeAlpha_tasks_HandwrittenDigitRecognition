import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Create folders
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Normalize the images
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Add image channel
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Show some training images
plt.figure(figsize=(8, 5))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.title("Digit: " + str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/sample_images.png")
plt.close()

# Create CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128
)

# Test the model
loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# Make predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Confusion matrix
cm = confusion_matrix(y_test, predicted_labels)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks(range(10))
plt.yticks(range(10))

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# Accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("results/accuracy_curve.png")
plt.close()

# Show some predictions
plt.figure(figsize=(10, 6))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(
        "Actual: " + str(y_test[i]) +
        "\nPredicted: " + str(predicted_labels[i])
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/example_predictions.png")
plt.close()

# Save the model
model.save("models/mnist_cnn.keras")

print("\nProject completed!")
print("Model saved in models folder.")
print("Results saved in results folder.")
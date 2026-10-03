"""
Handwritten Digit Recognition using CNN
Dataset: MNIST (loaded automatically through Keras)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load MNIST: 60,000 training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training data:", x_train.shape)
print("Test data:", x_test.shape)

# 2. Normalize pixels from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (samples, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 3. Display sample images
plt.figure(figsize=(8, 5))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/sample_images.png", dpi=150)
plt.close()

# 4. Build CNN model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.30),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.10,
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")

# 7. Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=150)
plt.close()

# 8. Predictions and classification report
y_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("MNIST Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(10))
plt.yticks(range(10))

threshold = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

# 10. Show example predictions
plt.figure(figsize=(10, 6))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title(f"Actual: {y_test[i]}\nPred: {y_pred[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/example_predictions.png", dpi=150)
plt.close()

# 11. Save model
model.save("models/mnist_cnn.keras")

print("\nProject completed.")
print("Model saved to: models/mnist_cnn.keras")
print("Results saved in: results/")

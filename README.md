# Handwritten Digit Recognition Using CNN

## 1. Project Overview
This project recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN).

The project uses the MNIST dataset. Keras provides MNIST directly through `keras.datasets.mnist.load_data()`.

## 2. Dataset
Official dataset/documentation:
https://keras.io/api/datasets/mnist/

MNIST contains 60,000 training images and 10,000 test images. Each image is a 28×28 grayscale image.

## 3. Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## 4. Folder Structure
```text
CodeAlpha_HandwrittenDigitRecognition
│
├── models
│   └── mnist_cnn.keras
│
├── results
│   ├── sample_images.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── example_predictions.png
│
├── train.py
├── README.md
└── requirements.txt
```

## 5. Installation
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
```

## 6. Run
```bash
python src/train.py
```

The MNIST dataset downloads automatically the first time the program runs.

## 7. Output
The program produces:
- Sample images
- Accuracy curve
- Confusion matrix
- Example predictions
- Trained CNN model

## 8. Important Note
Do not invent an accuracy value in the report. Run the program and copy the actual test accuracy shown in the terminal.

## 9. Internship Task
CodeAlpha Machine Learning Task 3 — Handwritten Character Recognition.

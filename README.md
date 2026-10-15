# CodeAlpha_HandwrittenCharacterRecognition

Handwritten digit recognition with a Convolutional Neural Network (CodeAlpha ML Internship, Task 3).

## Dataset
MNIST: 70,000 grayscale 28x28 images of handwritten digits 0-9 (60,000 train, 10,000 test).
54,000 images are used for training and 6,000 for validation (split from the training set).

## Model
CNN built with TensorFlow / Keras:
- Data augmentation (small random rotation, shift, zoom), active only during training
- 2 x [Conv(32) + BatchNorm] -> MaxPool -> Dropout
- 2 x [Conv(64) + BatchNorm] -> MaxPool -> Dropout
- Dense(128) -> Dropout(0.4) -> Dense(10, softmax)
- Adam optimizer, EarlyStopping and ReduceLROnPlateau callbacks

## Results (10,000 test images)
| Metric | Value |
|---|---|
| Test accuracy | 99.38% |
| Test loss | 0.0174 |
| Misclassified test images | 62 of 10,000 |
| Precision / recall / F1 (macro avg) | 0.9938 / 0.9938 / 0.9938 |

Every digit has precision, recall and F1 above 0.98. Digit 7 has the lowest precision (0.983) and digit 9 the lowest recall (0.988).

Most common mistakes (true -> predicted): 9 -> 4 (7 times), 5 -> 3 (5), 2 -> 7 (5), 1 -> 7 (5), 9 -> 7 (4).

A bold computer-font image of a 7 was also tested with `predict_image()` and predicted as 7 (confidence 0.96).

## What the notebook shows
1. Sample images and class distribution
2. Training and validation accuracy / loss curves
3. Classification report and confusion matrix
4. Misclassified examples
5. `predict_image()` function to test your own digit image

## Limitations
MNIST digits are clean and centered, so a custom photo may need a clear dark digit on a plain background.
This project recognizes single digits only. It can be extended to letters (EMNIST) or full words (CRNN).

## Run
Easiest: open `handwritten_recognition.ipynb` in Google Colab (Runtime > Change runtime type > GPU) and Run all.

Locally (needs Python 3.10-3.13, TensorFlow does not support 3.14 yet):
```
pip install -r requirements.txt
jupyter notebook handwritten_recognition.ipynb
```
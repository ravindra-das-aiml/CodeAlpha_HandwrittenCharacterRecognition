# %% [markdown]
# # Handwritten Digit Recognition with a CNN (CodeAlpha ML Internship - Task 3)
# Goal: recognize handwritten digits (0-9) from 28x28 grayscale images.
# Dataset: MNIST (70,000 images). Model: Convolutional Neural Network (TensorFlow / Keras).
# Run on Google Colab (Runtime > Change runtime type > GPU) or on a machine with TensorFlow installed.

# %%
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
EPOCHS = 15
print("TensorFlow:", tf.__version__)
print("GPU available:", bool(tf.config.list_physical_devices("GPU")))

# %% [markdown]
# ## 1. Load data

# %%
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train_full.shape, "Test:", x_test.shape)
print("Pixel range:", x_train_full.min(), "-", x_train_full.max())

# %%
fig, axes = plt.subplots(2, 10, figsize=(15, 3.6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_train_full[i], cmap="gray"); ax.set_title(int(y_train_full[i])); ax.axis("off")
plt.suptitle("Sample training images"); plt.tight_layout(); plt.show()

pd.Series(y_train_full).value_counts().sort_index().plot(
    kind="bar", figsize=(7, 3), title="Class distribution (training set)"); plt.show()

# %% [markdown]
# ## 2. Preprocessing
# Scale pixels to [0, 1], add a channel dimension, and hold out 10% of training data for validation.

# %%
def prep(x):
    return (x.astype("float32") / 255.0)[..., np.newaxis]

x_train_full, x_test = prep(x_train_full), prep(x_test)
n_val = 6000
idx = np.random.permutation(len(x_train_full))
x_val, y_val = x_train_full[idx[:n_val]], y_train_full[idx[:n_val]]
x_train, y_train = x_train_full[idx[n_val:]], y_train_full[idx[n_val:]]
print("train / val / test:", x_train.shape[0], x_val.shape[0], x_test.shape[0])

# %% [markdown]
# ## 3. CNN model
# Small random rotation, shift and zoom (data augmentation, active only during training)
# help the model handle different handwriting styles.

# %%
def build_model():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.06), layers.RandomTranslation(0.08, 0.08), layers.RandomZoom(0.08),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %% [markdown]
# ## 4. Train
# EarlyStopping and ReduceLROnPlateau stop the training when it no longer improves.

# %%
callbacks = [
    keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, monitor="val_loss"),
]
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=EPOCHS, batch_size=128, callbacks=callbacks, verbose=2)

# %%
h = pd.DataFrame(history.history)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
h[["accuracy", "val_accuracy"]].plot(ax=ax[0], title="Accuracy", marker="o")
h[["loss", "val_loss"]].plot(ax=ax[1], title="Loss", marker="o")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Evaluation on the test set

# %%
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}")
proba = model.predict(x_test, verbose=0)
y_pred = proba.argmax(axis=1)
print(classification_report(y_test, y_pred, digits=4))

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix (test set)"); plt.show()

off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted([(off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j] > 0], reverse=True)[:5]
print("Most common mistakes (count, true -> predicted):")
for n, t, p in pairs:
    print(f"  {n:3d}   {t} -> {p}")

# %% [markdown]
# ## 6. Misclassified examples

# %%
wrong = np.where(y_pred != y_test)[0]
print(f"{len(wrong)} of {len(y_test)} test images misclassified")
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for ax, i in zip(axes.ravel(), wrong[:12]):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title(f"true {y_test[i]}, pred {y_pred[i]}\nconf {proba[i].max():.2f}", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Predict on your own handwriting
# Write a digit in black on white paper (or white on black), take a photo or screenshot, upload it,
# and call `predict_image("your_file.png")`. The function converts it to a 28x28 MNIST-style image.

# %%
from PIL import Image, ImageOps

def predict_image(path, show=True):
    img = Image.open(path).convert("L")
    if np.array(img).mean() > 127:          # white background -> invert to match MNIST
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    bbox = img.getbbox()                     # crop to the digit and pad to a square
    if bbox:
        img = img.crop(bbox)
    side = max(img.size)
    square = Image.new("L", (int(side * 1.4), int(side * 1.4)), 0)
    square.paste(img, ((square.size[0] - img.size[0]) // 2, (square.size[1] - img.size[1]) // 2))
    arr = np.array(square.resize((28, 28), Image.LANCZOS), dtype="float32") / 255.0
    p = model.predict(arr[np.newaxis, ..., np.newaxis], verbose=0)[0]
    if show:
        plt.imshow(arr, cmap="gray"); plt.axis("off")
        plt.title(f"Prediction: {p.argmax()} (confidence {p.max():.2f})"); plt.show()
    return int(p.argmax()), float(p.max())

# In Google Colab, upload a file first:
# from google.colab import files; uploaded = files.upload(); predict_image(list(uploaded)[0])

# %% [markdown]
# ## 8. Save model

# %%
model.save("mnist_cnn_model.keras")
print("Saved mnist_cnn_model.keras")

# %% [markdown]
# ## Conclusion
# Write here: final test accuracy, which digits were confused most, and ideas for improvement.

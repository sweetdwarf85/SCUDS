import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def load_and_preprocess_data(base_path, target_size=(256, 256)):
    """Loads and preprocesses image data for binary classification."""
    data = []
    for condition in ["bleeding", "ischemia", "no_stroke"]:
        condition_path = os.path.join(base_path, condition)

        if condition == "no_stroke":
            for filename in os.listdir(condition_path):
                if filename.endswith(".png"):
                    img_path = os.path.join(condition_path, filename)
                    img = Image.open(img_path).convert("L").resize(target_size)
                    data.append((np.array(img, dtype=np.float32) / 255.0, 0))
        else:
            png_path = os.path.join(condition_path, "PNG")
            for filename in os.listdir(png_path):
                if filename.endswith(".png"):
                    img_path = os.path.join(png_path, filename)
                    img = Image.open(img_path).convert("L").resize(target_size)
                    data.append((np.array(img, dtype=np.float32) / 255.0, 1))
    return data

def create_dataset(data, batch_size=32, shuffle=True):
    images, labels = zip(*data)
    images = np.expand_dims(np.array(images), axis=-1)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_cnn(input_shape=(256, 256, 1)):
    """Builds a CNN model for binary classification."""
    inputs = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def train_model(model, train_dataset, val_dataset, epochs=20, model_save_path="best_cnn_model.keras"):
    callbacks = [
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]
    return model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)

def evaluate_model(model, test_dataset):
    results = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {results[0]:.4f}\nTest Accuracy: {results[1]:.4f}")
    return results

def calculate_f1_score(model, dataset):
    all_true_labels, all_predicted_labels = [], []
    for images, labels in dataset:
        predictions = (model.predict(images) > 0.5).astype(np.int32)
        all_true_labels.extend(labels.numpy().astype(np.int32))
        all_predicted_labels.extend(predictions.flatten())
    return f1_score(all_true_labels, all_predicted_labels, average="weighted")

if __name__ == "__main__":
    base_data_path = "data"
    target_size, batch_size, epochs = (256, 256), 32, 20
    all_data = load_and_preprocess_data(base_data_path, target_size)
    if not all_data:
        raise ValueError("No data loaded. Check your data path.")
    train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=[x[1] for x in all_data], random_state=42)
    train_dataset, test_dataset = create_dataset(train_data, batch_size), create_dataset(test_data, batch_size, shuffle=False)
    model_path = "best_cnn_model.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Loaded trained model.")
    else:
        model = build_cnn(input_shape=(target_size[0], target_size[1], 1))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        train_model(model, train_dataset, test_dataset, epochs=epochs)
    evaluate_model(model, test_dataset)
    print(f"Test F1 Score: {calculate_f1_score(model, test_dataset):.4f}")

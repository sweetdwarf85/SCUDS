import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def load_and_preprocess_data(base_path, target_size=(256, 256)):
    """Loads and preprocesses image data for binary classification."""
    data = []
    for condition in ["bleeding", "ischemia", "no_stroke"]:
        condition_path = os.path.join(base_path, condition)

        if condition == "no_stroke":
            for filename in os.listdir(condition_path):
                if filename.endswith(".png"):
                    patient_id = int(filename.split(".")[0])
                    img_path = os.path.join(condition_path, filename)
                    img = Image.open(img_path).convert("L")  
                    img = img.resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0  
                    data.append((img_array, 0, patient_id))  # label = 0 (inme olmamış)

        else:  # bleeding or ischemia(kanama veya iskemi)
            png_path = os.path.join(condition_path, "PNG")  # png kullan
            for filename in os.listdir(png_path):
                if filename.endswith(".png"):
                    patient_id = int(filename.split(".")[0])
                    img_path = os.path.join(png_path, filename)  # Correct path

                    img = Image.open(img_path).convert("L")
                    img = img.resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0

                    data.append((img_array, 1, patient_id))  # label = 1 (inme olmuş)

    return data

def create_dataset(data, batch_size=32, shuffle=True):
    """Creates a TensorFlow dataset."""
    images = []
    labels = []
    for img_array, label, _ in data:  # patient_id'ye trainleme kıosmında ihtiyacımız yok
        images.append(img_array)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # 1 dimension daha ekliyor: (N, H, W) -> (N, H, W, 1)
    images = np.expand_dims(images, axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))

    dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def data_augmentation(image, label):
    """Applies data augmentation."""

    # yatay çevirme rastgele şekilde
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

    # küçük derecelerde çeviriyor.
    angle = tf.random.uniform(shape=[], minval=-0.2, maxval=0.2)
    image = tf.image.rot90(image, k=tf.cast(angle * 2 / np.pi, tf.int32))

    return image, label


# İkili Sınıflandırma için CNN Modeli 

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
    outputs = Dense(1, activation='sigmoid')(x)  # 1 norön, sigmoid aktifleştirme

    model = Model(inputs=[inputs], outputs=[outputs])
    return model



def train_model(model, train_dataset, val_dataset, epochs=20,  
                model_save_path="best_cnn_model.keras"):
    """Trains the CNN model."""

    callbacks = [
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1), 
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1) 
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

def evaluate_model(model, test_dataset):
    """Evaluates the model."""
    results = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    return results


# --- ana script'in yazılı olduğu yer ---

if __name__ == "__main__":
    base_data_path = "data"  # dataset'leri, 
    target_size = (256, 256)
    batch_size = 32
    epochs = 20 

    all_data = load_and_preprocess_data(base_data_path, target_size)
    print(f"Total number of samples: {len(all_data)}")

    # boş data setini kontrol et
    if not all_data:
        raise ValueError("No data loaded. Check your data path and directory structure.")
    # Verileri bölme (ikili sınıflandırma için etikete göre katmanlandırma)
    train_data, test_data = train_test_split(
        all_data, test_size=0.2, random_state=42, stratify=[x[1] for x in all_data]
    )

    train_dataset = create_dataset(train_data, batch_size=batch_size, shuffle=True)
    test_dataset = create_dataset(test_data, batch_size=batch_size, shuffle=False)

    # cnn model kapanısı yani tamamlanma kismi
    model = build_cnn(input_shape=(target_size[0], target_size[1], 1))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']) # correct loss 

    # model train
    history = train_model(model, train_dataset, test_dataset, epochs=epochs)

    # model değerlendirme ksm
    evaluate_model(model, test_dataset)

    # F1 ölçümü hala bir sıkıntı yaşıyoruz
    def calculate_f1_score(model, dataset):
        all_true_labels = []
        all_predicted_labels = []

        for images, labels in dataset: # unpackliyor
            predictions = model.predict(images)
            predicted_labels_batch = (predictions > 0.5).astype(np.int32) 
            true_labels_batch = labels.numpy().astype(np.int32)
            all_true_labels.extend(true_labels_batch.flatten())
            all_predicted_labels.extend(predicted_labels_batch.flatten())
        return tf.keras.metrics.F1Score()(all_true_labels, all_predicted_labels).numpy()

    test_f1 = calculate_f1_score(model, test_dataset)
    print(f"Test F1 Score: {test_f1}")
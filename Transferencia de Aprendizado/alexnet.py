import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "final_dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "final_dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "final_dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

def preprocess(ds):
    return ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

train_ds = preprocess(train_ds)
val_ds = preprocess(val_ds)
test_ds = preprocess(test_ds)

def build_alexnet(input_shape, num_classes, weight_decay):
    model = models.Sequential([
        tf.keras.Input(input_shape),
        layers.Conv2D(96, kernel_size=(11,11), strides=4, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=2),

        layers.Conv2D(256, kernel_size=(5,5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=2),

        layers.Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(384, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3,3), strides=2),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))
    ])
    return model

num_classes = 4
epochs = 2
learning_rate = 0.001
weight_decay = 0.00001

IMG_SIZE = (224, 224)

model = build_alexnet(input_shape=IMG_SIZE + (3,), num_classes=num_classes, weight_decay=weight_decay)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Acurácia no conjunto de teste: {test_acc:.2%}")

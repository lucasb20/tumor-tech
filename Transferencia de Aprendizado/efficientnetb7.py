import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
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

num_classes = 4
epochs = 2
learning_rate = 0.0005
weight_decay = 0.0001
top_dropout_rate = 0.2

inputs = layers.Input(shape=IMG_SIZE + (3,))
model = EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")

model.trainable = False

x = layers.GlobalAveragePooling2D()(model.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(top_dropout_rate)(x)
outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(weight_decay))(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Acurácia no conjunto de teste: {test_acc:.2%}")

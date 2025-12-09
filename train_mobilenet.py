import os, json
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE   = (224, 224)
BATCH      = 32
EPOCHS_HEAD= 12       # train classification head
EPOCHS_FT  = 8        # fine-tune last layers
DATA_DIR   = "dataset"
MODEL_BEST = "smart_bin_mbv2_best.keras"
MODEL_FINAL= "smart_bin_mbv2.h5"
CLASSMAP   = "class_map.json"

# our 5 classes
CLASSES = ["glass", "metal", "paper", "plastic", "general"]

def save_classmap(gen):
    class_indices = gen.class_indices
    idx_to_class  = {v:k for k,v in class_indices.items()}
    with open(CLASSMAP, "w") as f:
        json.dump(idx_to_class, f)
    return idx_to_class

def build_gens():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.20,
        rotation_range=15, width_shift_range=0.10, height_shift_range=0.10,
        zoom_range=0.10, horizontal_flip=True
    )
    train = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', subset='training', shuffle=True, classes=CLASSES
    )
    val = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', subset='validation', shuffle=False, classes=CLASSES
    )
    return train, val

def build_model(num_classes):
    base = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                       include_top=False, weights="imagenet")
    base.trainable = False  # freeze for head training

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(base.input, out)
    return model, base

def class_weights(gen):
    counts = Counter(gen.classes)
    mx = max(counts.values())
    return {cls: mx/c for cls, c in counts.items()}

if __name__ == "__main__":
    # 1) build generators and save class map
    train, val = build_gens()
    idx_to_class = save_classmap(train)
    num_classes  = len(idx_to_class)

    # 2) build model
    model, base = build_model(num_classes)

    # 3) callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_BEST, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    cw = class_weights(train)

    # 4) train classification head
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history_head = model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS_HEAD,
        class_weight=cw,
        callbacks=callbacks
    )

    # 5) fine-tune last ~40 layers
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history_ft = model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS_FT,
        class_weight=cw,
        callbacks=callbacks
    )

    # 6) combine histories and save to JSON
    history_all = {}
    for key in history_head.history.keys():
        # concatenate the lists from the two training phases
        history_all[key] = history_head.history.get(key, []) + history_ft.history.get(key, [])

    with open("history.json", "w") as f:
        json.dump(history_all, f)
    print("Saved training history to history.json")

    # 7) save final model
    model.save(MODEL_FINAL)
    print(f"Saved: {MODEL_FINAL}  (best checkpoint: {MODEL_BEST})")

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, MaxPool2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Check if GPU is available and set it as the default device
# if tf.config.experimental.list_physical_devices('GPU'):
#     print('Using GPU')
#     tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
#     tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[0], 'GPU')
# else:
#     print('Using CPU')

# Load Google Drive
# from google.colab import drive
# drive.mount("/content/drive/")

def plot_curves(history):

    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    fig.set_size_inches(12,4)

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("screenshots/vgg16_model_plot.png")

    # loss = history.history["loss"]
    # val_loss = history.history["val_loss"]

    # accuracy = history.history["accuracy"]
    # val_accuracy = history.history["val_accuracy"]

    # epochs = range(len(history.history["loss"]))

    # # plt.figure(figsize=(15,5))

    # #plot loss
    # # plt.subplot(1, 2, 1)
    # plt.plot(epochs, loss, label = "training_loss")
    # plt.plot(epochs, val_loss, label = "val_loss")
    # plt.ylabel("Loss")
    # plt.xlabel("epochs")
    # plt.legend()
    # plt.savefig("screenshots/vgg16_cnn_loss_vs_epochs.png")
    # plt.clf()
    # # plt.show()

    # #plot accuracy
    # # plt.subplot(1, 2, 2)
    # plt.plot(epochs, accuracy, label = "training_accuracy")
    # plt.plot(epochs, val_accuracy, label = "val_accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("epochs")
    # plt.legend()
    # plt.savefig("screenshots/vgg16_cnn_accuracy_vs_epochs.png")
    # # plt.show()

# filepath
training_data_filepath = 'human_emotion_training_data/train'
validation_data_filepath = 'human_emotion_training_data/test'

# Init image data generator with rescale
train_data_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all training images
train_generator = train_data_gen.flow_from_directory(
    training_data_filepath,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True,
    subset='training', 
)

# Preprocess all testing images
validation_generator = validation_data_gen.flow_from_directory(
    validation_data_filepath,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# building a VGG16 model
emotion_model = Sequential()
emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))

emotion_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))

emotion_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))

emotion_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))

emotion_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(4096, activation="relu"))
emotion_model.add(Dense(7,activation="softmax"))

# print(emotion_model.summary())

# Creating model structure
# emotion_model = Sequential()

# emotion_model.add(Input(shape=(48, 48, 1))) # 1 because only 1 color channel as input images are greyscale

# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu')) 
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25)) # avoid overfitting

# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax')) # 7 because 7 emotions for output

# emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Creating model structure
# emotion_model = Sequential()

# emotion_model.add(Input(shape=(48, 48, 1)))  # 1 because only 1 color channel as input images are greyscale

# # Block 1
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

# # Block 2
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

# # Block 3
# emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
# emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))  # 7 because 7 emotions for output

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

emotion_model.summary()

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64
)

# model_json = emotion_model.to_json()
# with open("emotion_model.json", "w") as json_file:
#     json_file.write(model_json)
 
# save trained model weight in .h5 file
# checkpoint_path = "emotion_model.h5"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# emotion_model.save(checkpoint_dir)

plot_curves(emotion_model_info)

# serialize model to JSON
model_json = emotion_model.to_json()
with open("models/VGG16_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
emotion_model.save_weights("models/VGG16_model.h5")
print("Saved model to disk")

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, model_from_json, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Check if GPU is available and set it as the default device
# if tf.config.experimental.list_physical_devices('GPU'):
#     print('Using GPU')
#     tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
#     tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[0], 'GPU')
# else:
#     print('Using CPU')

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
    plt.savefig("screenshots/resnet50_model_plot.png")

# Load Google Drive
# from google.colab import drive
# drive.mount("/content/drive/")

# filepath
training_data_filepath = 'human_emotion_training_data/train'
validation_data_filepath = 'human_emotion_training_data/test'

img_shape = 224

# Custom preprocessing function to rescale and convert to RGB
def preprocess_image(image):
    # Convert to RGB
    image_rgb = np.stack((image[:, :, 0],) * 3, axis=-1)
    return image_rgb

# Init image data generator with custom preprocessing
train_data_gen = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,                                        
        fill_mode='nearest',
    )
validation_data_gen = ImageDataGenerator(rescale=1./255)  #, preprocessing_function=preprocess_image)


# Preprocess all training images
train_generator = train_data_gen.flow_from_directory(
    training_data_filepath,
    target_size=(img_shape, img_shape),
    batch_size=64,
    color_mode="rgb",
    class_mode='categorical',
    shuffle=True,
    subset='training', 
)

# Preprocess all testing images
validation_generator = validation_data_gen.flow_from_directory(
    validation_data_filepath,
    target_size=(img_shape, img_shape),
    batch_size=64,
    color_mode="rgb",
    class_mode='categorical',
    shuffle=False
)

ResNet50V2 = tf.keras.applications.ResNet50V2(input_shape=(img_shape, img_shape, 3),
                                               include_top= False,
                                               weights='imagenet'
                                               )

# Freezing all layers except last 50

ResNet50V2.trainable = True

for layer in ResNet50V2.layers[:-50]:
    layer.trainable = False

def Create_ResNet50V2_Model():

    model = Sequential([
                      ResNet50V2,
                      Dropout(.25),
                      BatchNormalization(),
                      Flatten(),
                      Dense(64, activation='relu'),
                      BatchNormalization(),
                      Dropout(.5),
                      Dense(7,activation='softmax')
                    ])
    return model

ResNet50V2_Model = Create_ResNet50V2_Model()

print(ResNet50V2_Model.summary())

ResNet50V2_Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create Callback Checkpoint
checkpoint_path = "ResNet50V2_Model_Checkpoint"

Checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True)

# Create Early Stopping Callback to monitor the accuracy
Early_Stopping = EarlyStopping(monitor = 'val_accuracy', patience = 7, restore_best_weights = True, verbose=1)

# Create ReduceLROnPlateau Callback to reduce overfitting by decreasing learning
Reducing_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.2,
                                                  patience=2,
#                                                   min_lr=0.00005,
                                                  verbose=1)

callbacks = [Early_Stopping, Reducing_LR]

steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size

ResNet50V2_history = ResNet50V2_Model.fit(train_generator ,validation_data = validation_generator , epochs=30, batch_size=64,
                                         callbacks = callbacks, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

ResNet50V2_Score = ResNet50V2_Model.evaluate(validation_generator)

print("    Test Loss: {:.5f}".format(ResNet50V2_Score[0]))
print("Test Accuracy: {:.2f}%".format(ResNet50V2_Score[1] * 100))

plot_curves(ResNet50V2_history)

# serialize model to JSON
model_json = ResNet50V2_Model.to_json()
with open("models/ResNet50_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ResNet50V2_Model.save_weights("models/ResNet50_model.h5")
print("Saved model to disk")


# # Load pre-trained VGG16 model without the top (fully connected) layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# # Freeze the pre-trained layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom top layers for emotion prediction
# x = base_model.output
# x = Flatten()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(7, activation='softmax')(x)  # 7 emotions

# # Create the full model
# emotion_model = Model(inputs=base_model.input, outputs=predictions)

# # Compile the model
# emotion_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# print(emotion_model.summary())

# # Train the model
# emotion_model_info = emotion_model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.n // train_generator.batch_size,
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.n // validation_generator.batch_size
# )

# visualkeras.layered_view(emotion_model, to_file='vgg16_emotion_model_diagram.png') # write and show

# plot_curves(emotion_model_info)
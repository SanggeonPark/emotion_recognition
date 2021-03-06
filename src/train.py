import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

from preprocess import preprocess_train_data
from models import TrainData, getCNNModel
from defines import input_shape_size, input_shape_2D, num_classes, facial_emotion_recognition_model_path

def train():
    train_data_array = preprocess_train_data()

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for index in range(len(train_data_array)):
        train_images.append(train_data_array[index].data)
        train_labels.append(train_data_array[index].label_index)
        if (index % 4 == 0):
            test_images.append(train_data_array[index].data)
            test_labels.append(train_data_array[index].label_index)


    train_images = np.array(train_images)
    train_images = np.expand_dims(train_images, axis=3)
    train_labels = np.array(train_labels)

    test_images = np.array(test_images)
    test_images = np.expand_dims(test_images, axis=3)
    test_labels = np.array(test_labels)

    # get CNN model
    model = getCNNModel()

    # Compile the model.
    model.compile(
      'adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

    # Train the compiled model.
    try:
        model.fit(
          train_images,
          to_categorical(train_labels, num_classes = num_classes, dtype='uint8'),
          epochs=4,
          validation_data=(test_images, to_categorical(test_labels, num_classes = num_classes, dtype='uint8')),
        )
    except Exception as e:
        print(str(e))
        return

    # Save the trained model
    model.save_weights(facial_emotion_recognition_model_path)

if __name__ == "__main__":
    # execute only if run as a script
    train()

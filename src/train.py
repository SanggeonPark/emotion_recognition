import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

from preprocess import preprocess_train_data
from models import TrainData
from switcher import l2i_switcher
from defines import input_shape_2D, input_shape_3D, num_filters, filter_size, pool_size, emotion_labels


# Map helper
def image_from_data(obj):
    return np.reshape(obj.data, input_shape_2D)

def label_from_data(obj):
    return int(obj.label_number)


train_data_array = preprocess_train_data()
train_images = np.array(map(image_from_data, train_data_array))
train_labels = np.array(map(label_from_data, train_data_array), dtype='int')

train_images = np.expand_dims(train_images, axis=3)

# define model with Sequential
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=input_shape_3D),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the defined model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the compiled model.
model.fit(
  train_images,
  to_categorical(train_labels, num_classes=len(emotion_labels)),
  epochs=3,
)

# Save the trained model
model.save_weights('facial_expression_recognition.h5')

import declxml as xml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from defines import num_filters, filter_size, pool_size, input_shape_3D, num_classes

# Dictionary Structure
xml_preprocessor = xml.dictionary('annotation', [
    xml.string('filename'),
    xml.dictionary('size', [
        xml.integer('width'),
        xml.integer('height'),
        xml.integer('depth')
    ]),
    xml.array(xml.dictionary('object', [
        xml.string('name'),
        xml.dictionary('bndbox', [
            xml.integer('xmin'),
            xml.integer('ymin'),
            xml.integer('xmax'),
            xml.integer('ymax')
        ])
    ]), alias='objects')
])

def dictionary_from_xml_file_url(xml_file_url):
    return xml.parse_from_file(xml_preprocessor, xml_file_url)

class TrainData:
    def __init__(self, data, label_index):
        self.data = data # reshaped numpy array
        self.label_index = label_index

# Convolutional Neural Networks Model
def getCNNModel():
    # define model with Sequential
    return Sequential([
      Conv2D(num_filters, filter_size, input_shape=input_shape_3D),
      MaxPooling2D(pool_size=pool_size),
      Flatten(),
      Dense(num_classes, activation='softmax'),
    ])

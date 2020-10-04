### Path
xml_folder = 'facial_emotion_data/train/annotations'
train_image_folder = 'facial_emotion_data/train/img'
test_image_folder = 'facial_emotion_data/test/img'
facial_emotion_recognition_model_path = 'facial_emotion_recognition.h5'
output_folder = 'output/'

### Emotion Labels
emotion_labels = ['neutral', 'anger', 'surprise', 'smile', 'sad']

### Constants

# Shape
input_shape_size = 48
input_shape_2D = (input_shape_size, input_shape_size)
input_shape_3D = (input_shape_size, input_shape_size, 1)

# Classes
num_classes = len(emotion_labels)

# Filer and Pool
num_filters = 8
filter_size = 3
pool_size = 2

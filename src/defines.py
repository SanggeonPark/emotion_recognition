# Path
xml_folder = 'facial_emotion_data/train/annotations'
train_image_folder = 'facial_emotion_data/train/img'
test_image_folder = 'facial_emotion_data/test/img'
face_detection_model_path = 'face_detection_data/haarcascade_frontalface_default.xml'
emotion_recognition_model_path = 'facial_expression_recognition.h5'

# Constants
input_shape_size = 36
input_shape_2D = (input_shape_size, input_shape_size)
input_shape_3D = (input_shape_size, input_shape_size, 1)

num_filters = 8
filter_size = 3
pool_size = 2

#Emotion Labels
emotion_labels = ['neutral', 'anger', 'surprise', 'smile', 'sad']

# Path
xml_folder = "facial_emotion_data/train/annotations"
train_image_folder = "facial_emotion_data/train/img"
test_image_folder = "facial_emotion_data/test/img"

# Constants
input_shape_size = 28
input_shape_2D = (input_shape_size, input_shape_size)
input_shape_3D = (input_shape_size, input_shape_size, 1)

num_filters = 8
filter_size = 3
pool_size = 2

#Emotion Labels
emotion_labels = ['neutral', 'anger', 'surprise', 'smile', 'sad']

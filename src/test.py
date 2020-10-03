from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

from switcher import i2l_switcher
from models import getCNNModel
from defines import test_image_folder, input_shape_2D, input_shape_3D, face_detection_model_path, facial_emotion_recognition_model_path

face_detection = cv2.CascadeClassifier(face_detection_model_path)

model = getCNNModel()
model.load_weights(facial_emotion_recognition_model_path)

for f in listdir(test_image_folder): #load test image file names
    test_image_file_path = join(test_image_folder, f)
    print('Test Image: ', test_image_file_path)
    gray_img = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
    faces = face_detection.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=input_shape_2D, flags=cv2.CASCADE_SCALE_IMAGE)
    face_images = []
    for (x,y,w,h) in faces:
        cropped_img = gray_img[y:y+h, x:x+w]
        resized_img = cv2.resize(cropped_img, input_shape_2D, interpolation = cv2.INTER_AREA)
        face_images.append(np.asarray(resized_img.reshape(input_shape_2D), 'float32')/255.0)

    if (len(face_images) > 0):
        predictions = model.predict(face_images)
        print(np.argmax(predictions, axis=1))

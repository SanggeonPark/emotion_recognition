from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

from models import getCNNModel
from defines import emotion_labels, test_image_folder, input_shape_2D, facial_emotion_recognition_model_path

def test():
    frontface_model_path = 'face_detection_data/haarcascade_frontalface_default.xml'
    profileface_model_path = 'face_detection_data/haarcascade_profileface.xml'
    frontface_cascade = cv2.CascadeClassifier(frontface_model_path)
    profileface_cascade = cv2.CascadeClassifier(profileface_model_path)

    model = getCNNModel()
    model.load_weights(facial_emotion_recognition_model_path)

    for f in listdir(test_image_folder): #load test image file names
        test_image_file_path = join(test_image_folder, f)
        print(test_image_file_path)
        gray_img = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
        faces = frontface_cascade.detectMultiScale(gray_img, 1.1, 4)
        if (len(faces) == 0):
            faces = profileface_cascade.detectMultiScale(gray_img, 1.1, 4)
        if (len(faces) == 0):
            print('FAIL - No face found')
            continue

        face_images = []
        for (x,y,w,h) in faces:
            cropped_img = gray_img[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, input_shape_2D, interpolation = cv2.INTER_AREA)
            shaped_array = np.reshape(np.asarray(resized_img, 'float32')/255.0, input_shape_2D)
            face_images.append(shaped_array)

        face_images = np.asarray(face_images)
        face_images = np.expand_dims(face_images, axis=3)
        predictions = model.predict(face_images)
        values = np.argmax(predictions, axis=1)
        labels = [emotion_labels[value] for value in values]
        print('SUCCESS', labels)
if __name__ == "__main__":
    # execute only if run as a script
    test()

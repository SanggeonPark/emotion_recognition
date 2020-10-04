from os import listdir, mkdir, remove
from os.path import isfile, join, exists
import numpy as np
import cv2

from models import getCNNModel
from defines import emotion_labels, test_image_folder, output_folder, input_shape_2D, facial_emotion_recognition_model_path

def save_ouput(file_name, image, faces, labels):
    output_image = image
    color = (255, 0, 0)

    start_points = []
    for index in range(len(labels)):
        (x,y,w,h) = faces[index]
        start_point = (x, y)
        start_points.append(start_point)
        end_point = (x+w, y+h)
        output_image = cv2.rectangle(output_image, start_point, end_point, color, 1)
        output_image = cv2.putText(output_image,labels[index],start_point,cv2.FONT_HERSHEY_COMPLEX,1,color,2)

    output_file_path = join(output_folder, file_name)
    cv2.imwrite(output_file_path, output_image)
    cv2.waitKey()

def test(target_path):
    frontface_model_path = 'face_detection_data/haarcascade_frontalface_default.xml'
    profileface_model_path = 'face_detection_data/haarcascade_profileface.xml'
    frontface_cascade = cv2.CascadeClassifier(frontface_model_path)
    profileface_cascade = cv2.CascadeClassifier(profileface_model_path)

    model = getCNNModel()
    model.load_weights(facial_emotion_recognition_model_path)

    for f in listdir(target_path): #load test image file names
        test_image_file_path = join(test_image_folder, f)
        image = cv2.imread(test_image_file_path,cv2.IMREAD_UNCHANGED)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = frontface_cascade.detectMultiScale(image, 1.1, 5)
        if (len(faces) == 0):
            faces = profileface_cascade.detectMultiScale(image, 1.1, 5)
        if (len(faces) == 0):
            print('FAIL - No face found:', test_image_file_path)
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
        print('SUCCESS', labels, test_image_file_path)

        # draw rectangle on the face and print the emotion
        save_ouput(f, image, faces, labels)

if __name__ == "__main__":

    # clean up the output folder
    if (exists(output_folder) == False):
        mkdir(output_folder)
    else:
        for f in listdir(output_folder):
            remove(join(output_folder, f))

    test(test_image_folder)

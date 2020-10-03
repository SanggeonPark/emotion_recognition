from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

from models import dictionary_from_xml_file_url, TrainData
from defines import xml_folder, train_image_folder, input_shape_2D

def preprocess_train_data():
    train_data_array = []

    for f in listdir(xml_folder): #load xml file names
        xml_file_url = join(xml_folder, f)
        if isfile(xml_file_url): # if it's file
            dictionary = dictionary_from_xml_file_url(xml_file_url)
            gray_img = cv2.imread(join(train_image_folder,dictionary['filename']), cv2.IMREAD_GRAYSCALE)
            # print(join(imageFolder,dictionary['filename']))
            for object in dictionary['objects']:
                y1 = object['bndbox']['ymin']
                y2 = object['bndbox']['ymax']
                x1 = object['bndbox']['ymin']
                x2 = object['bndbox']['ymax']
                croped_img = gray_img[y1:y2, x1:x2]
                try:
                    resized_img = cv2.resize(croped_img, input_shape_2D, interpolation = cv2.INTER_AREA)
                except Exception as e:
                    # print(str(e))
                    pass
                data = TrainData(np.asarray(resized_img, 'float32')/255.0, object['name'])
                train_data_array.append(data)
    # print(train_data[0].data)
    return train_data_array
import os
import pickle
import dlib
import cv2
import numpy as np
from collections import OrderedDict


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])



def caffe_detection_to_dlib_rectangle(image, detection):
    (h,w) = image.shape[:2]
    box = detection*np.array([w,h,w,h])
    (startX, startY, endX, endY) = box.astype('int')
    return dlib.rectangle(startX, startY, endX, endY)

def dlib_full_object_detection_to_list(obj, dtype='int'):
    result = np.zeros((68,2), dtype=dtype)
    for i,point in enumerate(obj.parts()):
        result[i] = point.x, point.y 
    return result            
        
def list_subdirs(directory):
    result = []
    for root, dirs, files in  os.walk(directory):
        result = dirs
        break
    return [x for x in result if '.' not in x ]

def load_label(label_file = os.path.join('models','svm','SVMFaceLabelsEncoding.pickle')):
    print ('[INFO] loading labels...')
    return pickle.loads(open(label_file, 'rb').read())

def load_recognition_model(face_recognition_model_file = os.path.join('models','svm','SVMFaceRecognitionModel.pickle')):
    print ('[INFO] loading Recognition model...')
    return pickle.loads(open(face_recognition_model_file, 'rb').read())

def load_detection_model():
    print ('[INFO] loading Detection model...')
    return cv2.dnn.readNetFromCaffe(read_detection_prototxt(), os.path.join('models', 'caffe', 'res10_300x300_ssd_iter_140000.caffemodel'))

def read_detection_prototxt():
    return 'models/caffe/deploy.prototxt.txt'

def load_lanadmark_model():
    print ('[INFO] loading Landmark model...')
    return dlib.shape_predictor('models/dlib/shape_predictor_68_face_landmarks.dat')

def load_face_embedding_model():
    print('[INFO] loading face embedding model...')
    return cv2.dnn.readNetFromTorch('models/torch/nn4.small2.v1.t7')

def image_resize(img, new_size=(896,672)):
    if not isinstance(img,(np.ndarray)):
            img = cv2.imread(str(img))
    return cv2.resize(img, new_size)
    
    

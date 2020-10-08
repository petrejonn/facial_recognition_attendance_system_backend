import os
import pickle
import cv2
import dlib
import numpy as np
from sklearn.svm  import SVC
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from .utils import (caffe_detection_to_dlib_rectangle as cdtdr,
                   dlib_full_object_detection_to_list as dfodtl,
                   FACIAL_LANDMARKS_68_IDXS,
                   list_subdirs, load_detection_model,load_lanadmark_model,
                   load_face_embedding_model, image_resize)


"""Face Recognition Module"""




class FaceDetect:
    """Single Image Recognition"""
    def __init__(self, imagePath, detectionModel, confidence):
        self._confidence = confidence
        self._image = self._readImage(image_resize(imagePath))
        (self._height,self._width) = self._image.shape[:2]
        self._model = detectionModel
        self._processedImage = self._processImage(self._image)
        self._detect()
        
    def _processImage(self, img):
        return cv2.dnn.blobFromImage(cv2.resize(img, (300,300)),1.0,(230,230),(104.0,177.0,123.0))
    
    def _readImage(self, imagePath):
        if not isinstance(imagePath,(np.ndarray)):
            return cv2.imread(imagePath)
        return imagePath

    def _detect(self):
        """
        Desc: Detects all the face in the instance image file
        Args: None
        Return: None
        """
        print ('[INFO] Detecting Faces...')
        self._model.setInput(self._processedImage)
        self._detections = self._model.forward()
        mask = self._detections[:,:,:,2] > self._confidence 
        self._detections = self._detections[mask,...]
    
    def getDetections(self):
        """
        Desc: Use to access coordinate of faces detected in the instance image file
        Args: None
        Return: 2D numpy array of floats
        """
        return self._detections
    
    def printBoxOnFace(self):
        """
        Desc: Print Triangle and probability of predictions on the instance image file
        Args: None
        Return: None
        """
        print ('[INFO] Printing Detected Faces...')
        for i in range(len(self._detections)):
            box = self._detections[i,3:7]*np.array([self._width,self._height,self._width,self._height])
            (startX, startY, endX, endY) = box.astype('int')

            text = '{:.2f}%'.format(self._detections[i,2]*100)
            y = startY - 10 if startY-10 > 10 else startY + 10
            cv2.rectangle(self._image, (startX, startY),(endX, endY),(200,200,200),2)
            cv2.putText(self._image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,100),2)
    
    def getImage(self):
        """
        Desc: Used to access image file of the instance
        Args: None
        Return: image file of the instance in BGR format
        """
        return self._image
        


        

class FaceLandmark:
    """Single Image Landmarks Detection"""
    def __init__(self, imagePath, landmarkModel, detectionModel, confidence=.8):
        self._image = self._readImage(image_resize(imagePath))
        self._model = landmarkModel
        self._facesDetected = self._detectFace(imagePath, detectionModel, confidence)
        self._landMarks = np.zeros((self._facesDetected.shape[0],68,2), dtype='int')
        self._detect()
        
        
    def _readImage(self,imagePath):
        if not isinstance(imagePath,(np.ndarray)):
            return cv2.imread(imagePath)
        return imagePath
    
    def _detectFace(self, imagePath, detectionModel, confidence):
        faceDetector = FaceDetect(imagePath, detectionModel, confidence)
        return faceDetector.getDetections()
    
    def _detect(self):
        """
        Desc: Detects all the faces landmark in the instance image file
        Args: None
        Return: None
        """
        print ('[INFO] Detecting Lanmarks...')
        for (i, face) in enumerate(self._facesDetected[:,3:7]):
            self._landMarks[i] =   dfodtl(self._model(self._image,cdtdr(self._image,face)))
    

    def getDetections(self):
        """
        Desc: Use to access coordinate of the faces landmarks detected in the instance image file
        Args: None
        Return: 3D numpy array of floats
        """
        return self._landMarks
    
    def printLandmarkOnFace(self):
        """
        Desc: Print dots on the landmarks detected on the instance image file
        Args: None
        Return: None
        """
        print ('[INFO] Printing Detected Landmarks...')
        for i in range(len(self._landMarks)):
            for (x, y) in self._landMarks[i]:
                cv2.circle(self._image,(x,y),2, (0,255,0), -1)

    def getImage(self):
        """
        Desc: Used to access image file of the instance
        Args: None
        Return: image file of the instance in BGR format
        """
        return self._image
    
    
class FaceAlign:
    """ Aligns a single face on an Image """
    def __init__(self, imagePath, faceLandmark, alignedLeftEye=(.35,.35), alignedFaceWidth=256, alignedFaceHeight=None):
        self._image = self._readImage(image_resize(imagePath))
        self._faceLandmark = faceLandmark
        self._alignedLeftEye = alignedLeftEye
        self._alignedFaceWidth = alignedFaceWidth
        self._alignedFaceHeight = alignedFaceHeight
        
        if self._alignedFaceHeight is None:
            self._alignedFaceHeight = self._alignedFaceWidth
            
    def _readImage(self, imagePath):
        if not isinstance(imagePath,(np.ndarray)):
            return cv2.imread(imagePath)
        return imagePath
            
    def _getEyesPts(self):
        (leftEyeStart, leftEyeEnd) = FACIAL_LANDMARKS_68_IDXS['left_eye']
        (rightEyeStart, rightEyeEnd) = FACIAL_LANDMARKS_68_IDXS['right_eye']
        leftEyePts = self._faceLandmark[leftEyeStart:leftEyeEnd]
        rightEyePts = self._faceLandmark[rightEyeStart:rightEyeEnd]
        return (leftEyePts, rightEyePts)
            
    def align(self):
        """
        Desc: align the object
        Args: None
        Return: An aligned face Image
        """
        (leftEyePts, rightEyePts) = self._getEyesPts()
        leftEyeCenter = leftEyePts.mean(axis=0).astype('int')
        rightEyeCenter = rightEyePts.mean(axis=0).astype('int')
        # Angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        alignedRightEyeX = 1.0 - self._alignedLeftEye[0]
        # Determin scale of resulting Image
        distance = np.sqrt((dX ** 2) + (dY ** 2))
        alignedDistance = (alignedRightEyeX - self._alignedLeftEye[0])
        alignedDistance *= self._alignedFaceWidth
        scale = alignedDistance / distance
        # Compute Median point between the two eyes
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                     (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # Rotation Matrix for rotating and scaling the face
        rotationMatrix = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # Update the translation component of the matrix
        tX = self._alignedFaceWidth * 0.5
        tY = self._alignedFaceHeight * self._alignedLeftEye[1]
        rotationMatrix[0, 2] += (tX - eyesCenter[0])
        rotationMatrix[1, 2] += (tY - eyesCenter[1])
        # Apply the affine transformation
        (w, h) = (self._alignedFaceWidth, self._alignedFaceHeight)
        output = cv2.warpAffine(self._image, rotationMatrix, (w,h), flags=cv2.INTER_CUBIC)
        return output
    
    def alignToPath(self, path, fileName):
        """
        Desc: Aligns the object face image and save it to a directory
        Args: path--> Directory the aligned image should be saved
              fileName--> File name to store aligned Image
        Return: None
        """
        path = os.path.join(path,'aligned')
        if not os.path.exists(path):
            os.makedirs(path)
        file = os.path.join(path,fileName+'.jpg')
        img = self.align()
        cv2.imwrite(file,img)
        
        
class FacesAlign:
    """Aligns all the Faces in an image File"""
    def __init__(self, imagePath, faceLandmarks, alignedLeftEye=(.35,.35), alignedFaceWidth=256, alignedFaceHeight=None):
        self._imagePath = image_resize(imagePath)
        self._faceLandmarks = faceLandmarks
        self._alignedImages = []
    
    def align(self):
        """
        Desc: align the object
        Args: None
        Return: An aligned face Images
        """
        for landmark in self._faceLandmarks:
            singleFace = FaceAlign(self._imagePath,landmark)
            self._alignedImages.append(singleFace.align())
        return self._alignedImages
    
    def alignToPath(self, path, fileName):
        """
        Desc: Aligns the object faces and save it to a directory
        Args: path--> Directory the aligned image should be saved
              fileName--> File name to store aligned Image
        Return: None
        """
        for i, landmark in enumerate(self._faceLandmarks):
            singleFace = FaceAlign(self._imagePath,landmark)
            singleFace.alignToPath(path, fileName)
    
    
class DatasetAlign:
    """ Align all the Images within a directory """
    def __init__(self, datasetDir):
        self._landmarkModel = load_lanadmark_model()
        self._detectionModel = load_detection_model()
        self._align(datasetDir)
    
    def _align(self,datasetDir):
        usersName = list_subdirs(datasetDir)
        for userName in usersName:
            print('[INFO] Processing '+userName+' Photos')
            imagesDir = os.path.join(datasetDir,userName)
            images = Path(imagesDir).glob('*.jpg')
            for i, image in enumerate(images):
                fileName = userName + str(i)
                print('[INFO] aligning image: '+fileName)
                faceLandmark = FaceLandmark(image, self._landmarkModel, self._detectionModel)
                faceAlign = FacesAlign(image,faceLandmark.getDetections())
                faceAlign.alignToPath(imagesDir, fileName)
    

    
class FaceEmbedding:
    """
    Takes a face Image and returns its embeddings
    """
    def __init__(self, faceImage, faceEmbeddingModel):
        self._model = faceEmbeddingModel
        self._processedFace = self._processFace(faceImage)
        
    def _processFace(self, faceImage):
        return cv2.dnn.blobFromImage(faceImage, 1.0/255, (96,96),(0,0,0),swapRB=True, crop=False)        
    
    def extract(self):
        """
        Desc: extracts all the embeddings from the face image
        Args: None
        Return: 128 features embedded from the face image
        """
        self._model.setInput(self._processedFace)
        return self._model.forward()
    

class DatasetEmbed:
    """Extract and save to a .pickle file all the  Images within a directory"""
    def __init__(self, datasetDir, dest):
        self._embeddings = []
        self._labels = []
        self._embeddingModel = load_face_embedding_model()
        self._embed(datasetDir, dest)
        
    def _embed(self, datasetDir, dest):
        usersName = list_subdirs(datasetDir)
        for userName in usersName:
            imagesDir = os.path.join(datasetDir,userName,'aligned')
            imageFiles = Path(imagesDir).glob('*.jpg')
            for i, image in enumerate(imageFiles):
                img = cv2.imread(str(image))
                embedding = FaceEmbedding(img, self._embeddingModel).extract()
                self._embeddings.append(embedding.flatten())
                self._labels.append(userName)
        data = {'embeddings': self._embeddings,'labels': self._labels}
        file = open(os.path.join(dest,'embeddings.pickle'), 'wb')
        file.write(pickle.dumps(data))
        file.close()
                
            
    
class TrainModel:
    """Train an SVM classifier with face embeddings"""
    def __init__(self, record, outputModelsFile, outputLabelsFile):
        self._record = self._loadRecord(record)
        self._labelEncoder = LabelEncoder()
        self._labels = self._labelEncoder.fit_transform(self._record['labels'])
        self._train(outputModelsFile, outputLabelsFile)
        
    def _loadRecord(self, record):
        print('[INFO] Loading Face Record...')
        return pickle.loads(open(record, 'rb').read())
        
    def _train(self, outputModelsFile, outputLabelsFile):
        print('[INFO] training model...')
        model = SVC(C=1.0, kernel='linear', probability=True)
        model.fit(self._record['embeddings'], self._labels)
        modelFile = open(outputModelsFile, 'wb')
        modelFile.write(pickle.dumps(model))
        modelFile.close()
        labelFile = open(outputLabelsFile, 'wb')
        labelFile.write(pickle.dumps(self._labelEncoder))
        labelFile.close()
        
        
class FaceRecognizer:
    """Predict a face Image"""
    def __init__(self, faceImage,recognitionModel, labels, embeddingModel):
        self._recognitionModel = recognitionModel
        self._labels = labels
        self._faceEmbedding = self._faceEmbedding(faceImage, embeddingModel)
    
    def _faceEmbedding(self, faceImage, embeddingModel, ):
        return FaceEmbedding(faceImage, embeddingModel).extract()
    
    def predict(self):
        """
        Desc: predict whos face in on the image file
        Args: None
        return: (name, probability)
        """
        prediction = self._recognitionModel.predict_proba(self._faceEmbedding)[0]
        highestProbabilityIndex = np.argmax(prediction)
        probability = prediction[highestProbabilityIndex]
        name = self._labels.classes_[highestProbabilityIndex]
        return (name, probability)

    
class FacesRecognizer:
    """Recognize all the faces detected on an image File"""
    def __init__(self, imageFile,recognitionModel, labels, detectionModel, landmarkModel, embeddingModel):
        self._recognitionModel = recognitionModel
        self._embeddingModel = embeddingModel
        self._labels = labels
        faceLandmarks = FaceLandmark(imageFile, landmarkModel, detectionModel)
        faceLandmarks.printLandmarkOnFace()
        self._landmarks = faceLandmarks.getDetections()
        self._image = faceLandmarks.getImage()
        self._alignedFaces = FacesAlign(imageFile, self._landmarks).align()
        self._recognitions = []
        
    def _readImage(self, imageFile):
        if not isinstance(imageFile,(np.ndarray)):
            return cv2.imread(imageFile)
        return imageFile
    
    def recognize(self, treshHold = .5):
        """
        Desc: Recognize all the faces within the objects Image file
        Args: None
        Return: None
        """
        for i, face in enumerate(self._alignedFaces):
            recognition = FaceRecognizer(face, self._recognitionModel, self._labels, self._embeddingModel).predict()
            if recognition[1] >= treshHold:
                self._recognitions.append(recognition)
            
    def printName(self):
        """
        Desc: Prints the names of the people recongized on their respective forehead
        Args: None
        Return: none
        """
        for recognition, leftEye in zip(self._recognitions, self._landmarks[:,19]):
            cv2.putText(self._image, str(recognition[0]), (leftEye[0], leftEye[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    def getImage(self):
        """
        Desc: Returns the objects Image
        Args: None
        Returns: Image file
        """
        return self._image
        

    
class VideoFaceRecognizer:
    """Predics faces on an video Feed"""
    def __init__(self, recognitionModel, labels, videoSrc, detectionModel, landmarkModel, embeddingModel):
        self._recognitionModel = recognitionModel
        self._labels = labels
        self._embeddingModel = embeddingModel
        self._videoSrc = videoSrc
        self._detectionModel = detectionModel
        self._landmarkModel = landmarkModel

            
    def start(self):
        """
        Desc: starts the streaming and recognition process
        Args: None
        Return: None
        """
        stream = cv2.VideoCapture(self._videoSrc)
        while True:
            (grabed, frame) = stream.read()
            if not grabed:
                break
            facesRecognizer = FacesRecognizer(frame, self._recognitionModel, self._labels, self._detectionModel, self._landmarkModel, self._embeddingModel)
            facesRecognizer.recognize(.9)
            facesRecognizer.printName()
            frame = facesRecognizer.getImage()
            cv2.imshow('Feed', frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        stream.release()
        cv2.destroyAllWindows()

    
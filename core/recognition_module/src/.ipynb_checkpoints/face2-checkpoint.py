import os
import math
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
from multiprocessing import Process, Manager, cpu_count
import face_recognition
import cv2
import time
import numpy as np
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class Train:
    def __init__(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        self.train_dir = train_dir
        self._extract(verbose)
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(self.face_encodings))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)
        self._train(n_neighbors, knn_algo, model_save_path)
    
    
    def _extract(self, verbose):
        print("Extracting KNN classifier...")
        self.face_encodings = []
        self.labels = []

        # Loop through each person in the training set
        for class_dir in os.listdir(self.train_dir):
            if not os.path.isdir(os.path.join(self.train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(self.train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    self.face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    self.labels.append(class_dir)
        print("Extracting KNN classifier Complete...")
                    
                    
    def _train(self, n_neighbors, knn_algo, model_save_path):
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(self.face_encodings, self.labels)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
           
        
class Predict:
    def __init__(self, img, knn_clf=None, model_path=None, distance_threshold=0.6):
#         if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
#             raise Exception("Invalid image path: {}".format(img_path))

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)
        else:
            self.knn_clf = knn_clf
            
        self.img = img
        self.distance_threshold = distance_threshold
    
    def _find_faces(self):
#         self.img = face_recognition.load_image_file(self.img_path)
        self.face_locations = face_recognition.face_locations(self.img)

    def predict(self):
        self._find_faces()
        if len(self.face_locations) == 0:
            return []
        
        faces_encodings = face_recognition.face_encodings(self.img, known_face_locations=self.face_locations)
        
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(self.face_locations))]
        
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), self.face_locations, are_matches)]
    
    def drawPrediction(self):
        predictions = self.predict()
        pil_image = Image.fromarray(np.uint8(self.img))
        draw = ImageDraw.Draw(pil_image)
        
        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font
            name = name.encode("UTF-8")

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw
        return np.array(pil_image)
    
    
class VidFaceRec:
    def __init__(self, videoSrc, modelPath, threshold):
        with open(modelPath, 'rb') as f:
            self.model = pickle.load(f)
        self.videoSrc = videoSrc
        self.threshold = threshold
    def start(self):
        """
        Desc: starts the streaming and recognition process
        Args: None
        Return: None
        """
        stream = cv2.VideoCapture(self.videoSrc)
        while True:
            (grabed, bgr_frame) = stream.read()
            if not grabed:
                break
            frame = bgr_frame[:, :, ::-1]
            predictor = Predict(frame, knn_clf=self.model,distance_threshold=self.threshold)
            frame = predictor.drawPrediction()
            frame = frame[:, :, ::-1].copy()
            cv2.imshow('Feed', frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        stream.release()
        cv2.destroyAllWindows()
        
        
        
        
# Get next worker's id
def next_id(current_id):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1


# Get previous worker's id
def prev_id(current_id):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1


# A subprocess use to capture frames.
def capture(read_frame_list, video_src):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(video_src)
    video_capture.set(3, 640)  # Width of the frames in the video stream.
    video_capture.set(4, 480)  # Height of the frames in the video stream.
    video_capture.set(5, 16) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num)
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()


# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list):

    while not Global.is_exit:

        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num):
            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame_process[:, :, ::-1]
        cv2.flip(rgb_frame, 0)

        predictor = Predict(rgb_frame, knn_clf=Global.model,distance_threshold=0.4)
        frame_process = predictor.drawPrediction()[:, :, ::-1]

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num)


if __name__ == '__main__':

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()
    

    # Number of workers (subprocess use to process frames)
    worker_num = cpu_count()

    # Subprocess list
    p = []
    
    
    # Create a subprocess to capture frames
    video_src = 0
    p.append(Process(target=capture, args=(read_frame_list, video_src)))
    p[0].start()
    
    # Create an output movie file (make sure resolution/frame rate matches input video!)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     output_movie = cv2.VideoWriter('output.avi', fourcc, 20, (640,480))


    #load models
    with open('models/knn/trained_knn_model.clf', 'rb') as f:
        Global.model = pickle.load(f)

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list)))
        p[worker_id].start()

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / np.sum(fps_list)
#             print("fps: %.2f" % fps)

            # Calculate frame delay, in order to make the video look smoother.
            # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
            # Larger ratio can make the video look smoother, but fps will hard to become higher.
            # Smaller ratio can make fps higher, but the video looks not too smoother.
            # The ratios below are tested many times.
            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num)])
#             output_movie.write(write_frame_list[prev_id(Global.write_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    cv2.destroyAllWindows()
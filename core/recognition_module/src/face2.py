import os
import math
from sklearn import neighbors
import cv2
import os.path
import pickle
from PIL import Image, ImageDraw
from multiprocessing import Process, Manager, cpu_count
import face_recognition
import time
import numpy as np
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


class Train:
    def __init__(
        self,
        train_dir,
        model_save_path=None,
        n_neighbors=None,
        knn_algo="ball_tree",
        verbose=True,
    ):
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
            for img_path in image_files_in_folder(
                os.path.join(self.train_dir, class_dir)
            ):
                img = cv2.imread(img_path)
                image = cv2.resize(img, dsize=(600, 800), interpolation=cv2.INTER_CUBIC)
                # image = face_recognition.load_image_file(img_path)
                # image = np.array(Image.open(img_path).resize((640, 480)))
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there is no face (or too many faces) in a training image, skip the image.
                    if verbose:
                        print(
                            "Image {} not suitable for training: {}".format(
                                img_path,
                                "Didn't find a face"
                                if len(face_bounding_boxes) < 1
                                else "Found more than one face",
                            )
                        )
                else:
                    # Add face encoding for current image to the training set
                    self.face_encodings.append(
                        face_recognition.face_encodings(
                            image, known_face_locations=face_bounding_boxes
                        )[0]
                    )
                    self.labels.append(class_dir)
                    print("Done processing one image")

        print("Extracting KNN classifier Complete...")

    def _train(self, n_neighbors, knn_algo, model_save_path):
        knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
        )
        knn_clf.fit(self.face_encodings, self.labels)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, "wb") as f:
                pickle.dump(knn_clf, f)


class Predict:
    def __init__(self, img, knn_clf=None, model_path=None, distance_threshold=0.6):
        #         if not os.path.isfile(img_path) or os.path.splitext(img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        #             raise Exception("Invalid image path: {}".format(img_path))

        if knn_clf is None and model_path is None:
            raise Exception(
                "Must supply knn classifier either thourgh knn_clf or model_path"
            )

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, "rb") as f:
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
        else:
            print("Face Found")

        faces_encodings = face_recognition.face_encodings(
            self.img, known_face_locations=self.face_locations
        )

        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [
            closest_distances[0][i][0] <= self.distance_threshold
            for i in range(len(self.face_locations))
        ]

        return [
            (pred, loc) if rec else ("unknown", loc)
            for pred, loc, rec in zip(
                self.knn_clf.predict(faces_encodings), self.face_locations, are_matches
            )
        ]

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
            draw.rectangle(
                ((left, bottom - text_height - 10), (right, bottom)),
                fill=(0, 0, 255),
                outline=(0, 0, 255),
            )
            draw.text(
                (left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255)
            )

        # Remove the drawing library from memory as per the Pillow docs
        del draw
        return np.array(pil_image)

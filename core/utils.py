import os
import pickle
from core.recognition_module.src.face2 import Predict
from core.recognition_module.src.face2 import Train


def train_model():
    Train(
        "core/recognition_module/dataset",
        model_save_path="core/recognition_module/models/knn/trained_knn_model.clf",
    )


def recognize(imgFile):
    # load models
    with open("core/recognition_module/models/knn/trained_knn_model.clf", "rb") as f:
        knn_model = pickle.load(f)
    predictor = Predict(imgFile, knn_clf=knn_model, distance_threshold=0.4)
    return [prediction[0] for prediction in predictor.predict()]

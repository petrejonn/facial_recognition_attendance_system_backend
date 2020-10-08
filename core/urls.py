from django.urls import path
from django.contrib.auth import views as auth_view

from . import views

app_name = "core"
urlpatterns = [
    path("student/create", views.StudentCreateView.as_view(), name="student_create"),
    # path("train", views.TrainView.as_view(), name="trainning"),
    path("recognize", views.RecognitionView.as_view(), name="recognize"),
]

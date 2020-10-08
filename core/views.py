import os
import zipfile
import base64
import cv2
from django.core.files.base import ContentFile
from django.core.exceptions import ObjectDoesNotExist
from django.urls import reverse_lazy, reverse
from django.shortcuts import HttpResponseRedirect, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from core.utils import recognize
from .models import Student
from core.utils import train_model


class RecognitionView(APIView):
    def post(self, request, *args, **kwargs):
        format, imgstr = self.request.data.get("img").split(";base64,")
        ext = format.split("/")[-1]
        myfile = ContentFile(base64.b64decode(imgstr), name="temp." + ext)
        fs = FileSystemStorage()
        fileName = fs.save("temp." + ext, myfile)
        student = recognize(cv2.imread(fs.url(fileName)))
        os.remove(fs.url(fileName))
        if len(student) < 1:
            return Response("processing...")
        else:
            print("\n" + student[0] + "\n")
            return Response(student[0])


class StudentCreateView(APIView):
    def post(self, request, *args, **kwargs):
        file_name = request.data.get("reg_no").replace("/", "")
        std = Student(full_name=request.data.get("full_name"), reg_no=file_name,)
        std.save()
        myfile = self.request.FILES["photos"]
        fs = FileSystemStorage()

        filename = fs.save(file_name + ".zip", myfile)
        with zipfile.ZipFile(
            os.path.join(settings.MEDIA_URL, filename), "r"
        ) as zip_ref:
            zip_ref.extractall(
                os.path.join(
                    "core", os.path.join("recognition_module", "dataset/" + file_name)
                )
            )
            os.remove(os.path.join(settings.MEDIA_URL, filename))
        train_model()
        return Response("SUCCESSFULLY SAVED")

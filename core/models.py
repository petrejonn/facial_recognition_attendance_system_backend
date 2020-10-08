from django.db import models


class Student(models.Model):

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    full_name = models.CharField(max_length=255)
    reg_no = models.CharField(max_length=255)

    class Meta:
        """Meta definition for Student."""

        verbose_name = "Student"
        verbose_name_plural = "Students"

    def __str__(self):
        """Unicode representation of Student."""
        return self.full_name

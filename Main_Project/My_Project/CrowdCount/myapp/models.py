from django.db import models

class UploadedFile(models.Model):
    file_path = models.CharField(max_length=255, default='default_value_here')
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
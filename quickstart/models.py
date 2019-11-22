from django.db import models
import os
def get_image_path(instance, filename):
    return os.path.join('fingerprints', str(instance.id), filename)

class Language(models.Model):
    name = models.CharField(max_length=50)
    paradigm = models.CharField(max_length=50)
    fingerprint = models.ImageField(upload_to=get_image_path, blank=True, null=True)
    def __str__(self):
        return self.name



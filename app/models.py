from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Visitor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    view_password = models.CharField(max_length=255, help_text="Plain text password for admin viewing")

    def __str__(self):
        return self.user.username

class Guest(models.Model):
    added_by = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15)
    email = models.EmailField(null=True, blank=True)
    purpose = models.TextField()
    photo = models.ImageField(upload_to='visitor_photos/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class BlockedVisitor(models.Model):
    added_by = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15)
    email = models.EmailField(null=True, blank=True)
    reason = models.TextField(default="No reason provided")
    photo = models.ImageField(upload_to='blocked_photos/', null=True, blank=True)
    blocked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
class todaysvisiter(models.Model):
    visitername=models.CharField(max_length=100)
    dateofvisit=models.CharField(max_length=100)
    image=models.ImageField(upload_to="visited photo")
    status=models.CharField(max_length=100,null=True)
    emotion = models.CharField(max_length=100, null=True)   


from django.urls import path
from .views import health, predict, predict_video

urlpatterns = [
    path("health/", health),
    path("predict/", predict),
    path("predict-video/", predict_video),
]
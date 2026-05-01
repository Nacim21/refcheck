from django.urls import path

from . import views

urlpatterns = [
    path("api/create-upload-url", views.create_upload_url, name="create_upload_url"),
    path("", views.home, name="home"),
    path("analyze/", views.analyze, name="analyze"),
    path("result/", views.result, name="result"),
    path("about/", views.about, name="about"),
]

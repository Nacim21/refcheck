from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("analyze/", views.analyze, name="analyze"),
    path("result/", views.result, name="result"),
    path("about/", views.about, name="about"),
]

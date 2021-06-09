from django.urls import path, include
from .views import main, result
from .predict import Net

urlpatterns = [
    path('',main),
    path('res/',result),
]
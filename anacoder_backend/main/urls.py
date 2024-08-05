
from django.urls import path, include
from .views import *

urlpatterns = [
    path("llm/response/", process_query),
]

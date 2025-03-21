from django.urls import path
from . import views
urlpatterns = [
    path ('account/',views.account, name = 'account'),
    path('', views.chat, name='chat'),
    path('get_ai_response/', views.get_ai_response, name='get_ai_response'),
    # path('get_ai_response_stream/', views.get_ai_response_stream, name='get_ai_response_stream'),
]

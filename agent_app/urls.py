# agent_app/urls.py


from django.urls import path
from . import views

urlpatterns = [
    path('agent/', views.agent_api_view, name='agent_api'),
    path('history/', views.get_history_view, name='get_history'), # New URL for history
]
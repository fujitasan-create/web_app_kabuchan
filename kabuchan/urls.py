from django.urls import path
from . import views

app_name='kabuchan'
urlpatterns=[
    path("",views.index,name='index'),
    path('about/', views.about, name='about'), 
    path('recommend/', views.recommend, name='recommend'),
    path('predict/', views.predict, name='predict'),
    path("market-result/", views.market_result, name="market_result"),
]
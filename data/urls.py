from django.urls import path
from data import views

app_name = 'data'

urlpatterns =  [
    path("", views.HomePageView.as_view() , name='home'),
    path('graph/', views.ChartView.as_view(), name='graph'),
    path('predictions/', views.PredictionChartView.as_view(), name='predictions'),
    path('percentage-per-hour/', views.ChartView.hourChangePercentChart, name ='percentage-per-hour'),
    path('market-cap/', views.ChartView.MarketCapChart, name ='market-cap'),
    path('circulating-supply/', views.ChartView.CirculatingSupplyChart, name ='circulating-supply'),
    path('trading-volume/', views.ChartView.TradingVolumeChart, name ='trading-volume'),
    path('price/', views.ChartView.PriceChart, name ='price'),
    path('vol-and-price/', views.ChartView.CompareVolAndPrice, name ='vol-and-price'),
    path('market-and-price/', views.ChartView.CompareMarketAndPrice, name ='market-and-price'),
    # path('plotly-chart/', views.PlotlyChartView.as_view(), name='plotly-chart')
]
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView

# processing data
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import requests
from bs4 import BeautifulSoup
import re

# Evalution 
from sklearn.preprocessing import MinMaxScaler

# model building 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# PLotting
import matplotlib as pl
import matplotlib.pyplot as plt
pl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo


# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs): 
        queryset = Api.getData(request)
        sundayset = Api.getSundays(request)
        context = {'tableData': queryset, 'sundays': sundayset}
        return render(request, 'data/index.html', context)

def scrapeData(request):
        # StoringInfo variables
        name, marketCap, price, circulatingSupply, tradingVolume, hourPercent, dayPercent, weekPercent, symbol = [], [], [], [], [], [], [], [], []
            
        # get todays date
        today = date.today()

        #determine whether its a Sunday
        date_day = today.weekday()

        #if its not Sunday then get and set Sunday's date
        if(date_day != 6):
            today = today - timedelta(date_day + 1)

        if request.GET.get('sundays'):
            # change the date (Jan 03, 2023) into date format (2023-01-03)
            date_format = "%b. %d, %Y"
            date_object = datetime.strptime(request.GET.get('sundays'), date_format)
            # change the date into 20230103
            new_format = "%Y%m%d"
            today = date_object.strftime(new_format)
            #remove the dash of date format
            current_date = today.replace('-', '')
        else: 
            #remove the dash of date format
            current_date = today.isoformat().replace('-', '')
            
        # URL to scrape
        url = "https://coinmarketcap.com/historical/"  + current_date

        # Request a website
        webpage = requests.get(url)
        # parse the text
        soup = BeautifulSoup(webpage.text, 'html.parser')
            
        # Get table row element
        tr = soup.find_all('tr', attrs={'class':'cmc-table-row'})

        count = 0
            
        for row in tr:
            if count == 20:
                break
            else:
                count += 1
                    
                # Store name of the crypto currency            
                name_col = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sticky cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__name'})
                cryptoname = name_col.find('a', attrs={'class':'cmc-table__column-name--name cmc-link'}).text.strip()
                    
                #trading volume
                tradingvolume = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__volume-24-h'}).text.strip()
                    
                # Market cap
                marketcap = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__market-cap'}).text.strip()
                    
                # Price
                crytoprice = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__price'}).text.strip()
                    
                # Circulating supply and symbol            
                circulatingSupplySymbol = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__circulating-supply'}).text.strip()
                supply = circulatingSupplySymbol.split(' ')[0]
                sym = circulatingSupplySymbol.split(' ')[1]
                    
                #1h
                hourpercent = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__percent-change-1-h'}).text.strip()
                    
                #24h
                daypercent = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__percent-change-24-h'}).text.strip()
                    
                #7d
                weekpercent = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__percent-change-7-d'}).text.strip()
                    
                # append the data
                name.append(cryptoname)
                marketCap.append(marketcap)
                price.append(crytoprice)
                circulatingSupply.append(supply)
                tradingVolume.append(tradingvolume)
                hourPercent.append(hourpercent)
                dayPercent.append(daypercent)
                weekPercent.append(weekpercent)
                symbol.append(sym)  
                
        return name, marketCap, price, circulatingSupply, tradingVolume, hourPercent, dayPercent, weekPercent, symbol


class Api(TemplateView):
    def getData(request): 
        name, marketCap, price, circulatingSupply, tradingVolume, hourPercent, dayPercent, weekPercent, symbol = scrapeData(request)
        # Create dataframe
        df = pd.DataFrame()
        df['Name'] = name
        df['Market Cap ($)'] = marketCap 
        df['Price ($)'] = price
        df['Circulating Supply'] = circulatingSupply
        df['Trading Volume 24h ($)'] = tradingVolume
        df['1h (%)'] = hourPercent
        df['24h (%)'] = dayPercent
        df['7d (%)'] = weekPercent
        df['Symbol'] = symbol

        return df

    def getCleanedData(request): 
        name, marketCap, price, circulatingSupply, tradingVolume, hourPercent, dayPercent, weekPercent, symbol = scrapeData(request)
        # Create dataframe
        df = pd.DataFrame()
        df['Name'] = name
        df['Market Cap ($)'] = marketCap 
        df['Price ($)'] = price
        df['Circulating Supply'] = circulatingSupply
        df['Trading Volume 24h ($)'] = tradingVolume
        df['1h (%)'] = hourPercent
        df['24h (%)'] = dayPercent
        df['7d (%)'] = weekPercent
        df['Symbol'] = symbol

        # Clean data that have symbols
        for i in range(len(df)):
            df['Market Cap ($)'][i] = df['Market Cap ($)'][i] = float(re.sub(r'[$,]', '', df['Market Cap ($)'][i]))
            df['Price ($)'][i] = df['Price ($)'][i] = float(re.sub(r'[$,]', '', df['Price ($)'][i]))
            df['Circulating Supply'][i] = df['Circulating Supply'][i] = float(re.sub(r'[$,]', '', df['Circulating Supply'][i]))
            df['Trading Volume 24h ($)'][i] = df['Trading Volume 24h ($)'][i] = float(re.sub(r'[$,]', '', df['Trading Volume 24h ($)'][i]))
            df['1h (%)'][i] = df['1h (%)'][i] = float(re.sub(r'[%]', '', df['1h (%)'][i]))
            df['24h (%)'][i] = df['24h (%)'][i] = float(re.sub(r'[%]', '', df['24h (%)'][i]))
            df['7d (%)'][i] = df['7d (%)'][i] = float(re.sub(r'[%]', '', df['7d (%)'][i]))

        return df

    def getSundays(request):
        # Define the start and end dates
        start_date = date(datetime.now().year, 1, 1)
        end_date = datetime.now().date()

        # Define a list to store the dates of every Saturday
        sunday_dates = []

        # Get the dates of every Saturday between the start and end dates
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 6:  # 5 represents Saturday
                sunday_dates.append(current_date)
            current_date += timedelta(days=1)

        return sunday_dates
    
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

# from plotly.offline import plot
# import plotly.graph_objs as go

class ChartView(TemplateView):
    def get(self, request, *args, **kwargs):
        return render(request, 'charts/visualization.html')

    def hourChangePercentChart(request):
        x, y1, y2, y3 = [], [], [], []
        data = Api.getCleanedData(request)

        x = data['Symbol']
        y1 = data['1h (%)']
        y2 = data['24h (%)']
        y3 = data['7d (%)']
    
        fig, ax = plt.subplots(figsize=(15, 6)) 
        ax.plot(x,y1, label="Percentage Changes per hour")
        ax.plot(x,y2, label="Percentage Changes per day")
        ax.plot(x,y3, label="Percentage Changes per week")
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Pecentage (%)')
        ax.set_title('Percentage of Changes According to Per Hours, Per Days and Per Week')
        ax.legend()
        
        response = HttpResponse(content_type = "image/jpeg")
        fig.savefig(response, format = "jpg")
        return response

    def MarketCapChart(request):
        data = Api.getCleanedData(request)
        x, y = [], []

        data = data.sort_values('Market Cap ($)')

        x = data['Symbol']
        y = data['Market Cap ($)']

        fig1, ax1 = plt.subplots(figsize=(15, 6))
        ax1.bar(x,y)
        ax1.set_yscale('log')
        ax1.set_xlabel('Symbol')
        ax1.set_ylabel('Market Cap ($)')
        ax1.set_title('Market Cap of Cryptocurrencies')
        
        response = HttpResponse(content_type = "image/jpeg")
        fig1.savefig(response, format = "jpg")
        return response

    def CirculatingSupplyChart(request):
        x, y = [], []
        data = Api.getCleanedData(request)
        data = data.sort_values('Circulating Supply')

        x = data['Symbol']
        y = data['Circulating Supply']

        fig2, ax2 = plt.subplots(figsize=(15, 6))  
        ax2.plot(x, y)
        ax2.set_yscale('log')
        ax2.set_xlabel('Symbol')
        ax2.set_ylabel('Circulating Supply')
        ax2.set_title('Circulating Supply of Cryptocurrencies')
        
        response = HttpResponse(content_type = "image/jpeg")
        fig2.savefig(response, format = "jpg")
        return response
    
    def TradingVolumeChart(request):
        x, y = [], []
        data = Api.getCleanedData(request)
        data = data.sort_values('Trading Volume 24h ($)')

        x = data['Name']
        y = data['Trading Volume 24h ($)']
        total = 0

        for volume in y:
            total += volume

        fig3, ax3 = plt.subplots(figsize=(12, 6))  
        ax3.pie(y, labels=None)
        ax3.axis('equal')
        labels = [f'{l}, {(s/total)*100:0.1f}%' for l, s in zip(x, y)]
        ax3.legend(labels, loc='center left', bbox_to_anchor=(0.8, 0.5))
        ax3.set_title('Trading Volume (24h) of Cryptocurrencies')
        
        response = HttpResponse(content_type = "image/jpeg")
        fig3.savefig(response, format = "jpg")
        return response

    def PriceChart(request):
        x, y = [], []
        data = Api.getCleanedData(request)

        x = data['Symbol']
        y = data['Price ($)']

        fig4, ax4 = plt.subplots(figsize=(12, 6))  
        ax4.bar(x, y)
        ax4.set_yscale('log')
        ax4.set_xlabel('Symbol')
        ax4.set_ylabel('Price ($)')
        ax4.set_title('Prices of Cryptocurrencies')
        
        response = HttpResponse(content_type = "image/jpeg")
        fig4.savefig(response, format = "jpg")
        return response
        
    def CompareVolAndPrice(request):
        x, y, labels = [], [], []
        data = Api.getCleanedData(request)

        x = data['Trading Volume 24h ($)']
        y = data['Price ($)']

        fig5, ax5 = plt.subplots(figsize=(12, 6))  
        ax5.scatter(x, y)
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        ax5.set_ylabel('Price ($)')
        ax5.set_xlabel('Trading Volume 24h ($)')
        ax5.set_title('Trading Volume against Price')
        
        response = HttpResponse(content_type = "image/jpeg")
        fig5.savefig(response, format = "jpg")
        return response

    def CompareMarketAndPrice(request):
        x, y = [], []
        data = Api.getCleanedData(request)

        x = data['Market Cap ($)']
        y = data['Price ($)']

        fig6, ax6 = plt.subplots(figsize=(12, 6))  
        ax6.scatter(x, y)
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.set_ylabel('Price ($)')
        ax6.set_xlabel('Market Cap ($)')
        ax6.set_title('Market Cap against Price')
        
        response = HttpResponse(content_type = "image/jpeg")
        fig6.savefig(response, format = "jpg")
        return response

class PredictionChartView(TemplateView):
    def get(self, request, *args, **kwargs):
        # get data according to cryptocurrencies
        url = ""
        if(request.GET.get('crypto')):
            url = 'datasets/' + request.GET.get('crypto') + '.csv'
        else:
            url = "datasets/BTC-USD.csv"

        # get data
        maindf = pd.read_csv(url)
        maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')

        
        y_2020 = maindf.loc[(maindf['Date'] >= '2020-01-01')
                     & (maindf['Date'] < '2023-01-31')]
        
        # show the graph for the cryptocurrencies of its open, close, high and low price
        names = cycle(['Cryptocurrencies Open Price','Cryptocurrencies Close Price','Cryptocurrencies High Price','Cryptocurrencies Low Price'])
        fig1 = go.Figure()
        fig1 = px.line(y_2020, x=y_2020.Date, y=[y_2020['Open'], y_2020['Close'], 
                                                y_2020['High'], y_2020['Low']],
                    labels={'Date': 'Date','value':'Cryptocurrencies value'})
        fig1.update_layout(title_text='Cryptocurrencies analysis chart', font_size=15, font_color='black',legend_title_text='Cryptocurrencies Parameters')
        fig1.for_each_trace(lambda t:  t.update(name = next(names)))
        fig1.update_xaxes(showgrid=False)
        fig1.update_yaxes(showgrid=False)
        plot_div = pyo.plot(fig1, output_type='div')

        # shww the graph of whole period of timeframe of close price of the whole year
        fig2 = go.Figure()
        closedf = maindf[['Date','Close']]
        fig2 = px.line(closedf, x=closedf.Date, y=closedf.Close,labels={'date':'Date','close':'Close Cryptocurrencies'})
        fig2.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
        fig2.update_layout(title_text='Whole period of timeframe of close price', plot_bgcolor='white', 
                        font_size=15, font_color='black')
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=False)
        plot_div2 = pyo.plot(fig2, output_type='div')

        # get close price data
        closedf = closedf[closedf['Date'] > '2022-02-01']
        close_stock = closedf.copy()

        # evaluate the model
        del closedf['Date']
        scaler=MinMaxScaler(feature_range=(0,1))
        closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
        training_size=int(len(closedf)*0.60)

        # slipt into training and testing set
        train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

        time_step = 15
        X_train, y_train = Api.create_dataset(train_data, time_step)
        X_test, y_test = Api.create_dataset(test_data, time_step)

        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        # build model
        model=Sequential()
        model.add(LSTM(10,input_shape=(None,1),activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error",optimizer="adam")

        history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)

        # make predictions
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        # transform back to original form
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
        original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

        # shift train predictions for plotting
        look_back=time_step
        trainPredictPlot = np.empty_like(closedf)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

        # shift test prediction for plotting
        testPredictPlot = np.empty_like(closedf)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

        # graph to compare original price with predicted price
        names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
        plotdf = pd.DataFrame({'date': close_stock['Date'],
                            'original_close': close_stock['Close'],
                            'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                            'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

        fig3 = go.Figure
        fig3 = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                                plotdf['test_predicted_close']],
                    labels={'value':'Cryptocurrencies price','date': 'Date'})
        fig3.update_layout(title_text='Comparision between original close price vs predicted close price',
                        plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
        fig3.for_each_trace(lambda t:  t.update(name = next(names)))

        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=False)
        plot_div3 = pyo.plot(fig3, output_type='div')

        x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        # get the prediction dataset
        lst_output=[]
        n_steps=time_step
        i=0
        pred_days = 30
        while(i<pred_days):
            
            if(len(temp_input)>time_step):
                
                x_input=np.array(temp_input[1:])
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
            
                lst_output.extend(yhat.tolist())
                i=i+1
                
            else:
                
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                
                lst_output.extend(yhat.tolist())
                i=i+1

        last_days=np.arange(1,time_step+1)
        day_pred=np.arange(time_step+1,time_step+pred_days+1)

        temp_mat = np.empty((len(last_days)+pred_days+1,1))
        temp_mat[:] = np.nan
        temp_mat = temp_mat.reshape(1,-1).tolist()[0]

        last_original_days_value = temp_mat
        next_predicted_days_value = temp_mat

        last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
        next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

        # graph to compare last 15 days and predict next 30 days
        new_pred_plot = pd.DataFrame({
            'last_original_days_value':last_original_days_value,
            'next_predicted_days_value':next_predicted_days_value
        })

        names = cycle(['Last 15 days close price','Predicted next 30 days close price'])

        fig4 = go.Figure()
        fig4 = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                            new_pred_plot['next_predicted_days_value']],
                    labels={'value': 'Cryptocurrencies price','index': 'Timestamp'})
        fig4.update_layout(title_text='Compare last 15 days vs next 30 days',
                        plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

        fig4.for_each_trace(lambda t:  t.update(name = next(names)))
        fig4.update_xaxes(showgrid=False)
        fig4.update_yaxes(showgrid=False)
        plot_div4 = pyo.plot(fig4, output_type='div')


        # graph to show whole year of historical data with the predicted next 30 days
        lstmdf=closedf.tolist()
        lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
        lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

        names = cycle(['Close price'])

        fig5 = go.Figure()
        fig5 = px.line(lstmdf,labels={'value': 'Cryptocurrencies price','index': 'Timestamp'})
        fig5.update_layout(title_text='Plotting whole closing stock price with prediction',
                        plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Cryptocurrencies')

        fig5.for_each_trace(lambda t:  t.update(name = next(names)))

        fig5.update_xaxes(showgrid=False)
        fig5.update_yaxes(showgrid=False)
        plot_div5 = pyo.plot(fig5, output_type='div') 

        return render(request, 'charts/predictions.html', {'plot_div': plot_div, 'plot_div2': plot_div2, 'plot_div3': plot_div3, 'plot_div4': plot_div4, 'plot_div5': plot_div5})

    
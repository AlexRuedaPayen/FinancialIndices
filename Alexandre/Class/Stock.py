import pandas
import math
import datetime

class Stock:

    def __init__(self,name,locally_stored=False,historical=True):
        if (locally_stored):
            self.stock=pandas.read_csv(filepath_or_buffer='./Data/'+name+'.csv')
            self.stock.index=self.stock['Date']
            print(f'Data from '+ str(min(self.stock['Date'].values))+' to '+str(max(self.stock['Date'].values)))
        else:
            if (historical):
                import requests
                from bs4 import BeautifulSoup
                url=('https://uk.finance.yahoo.com/quote/'+name+'/history?p='+name)
                r=requests.get(url)
                web_content=BeautifulSoup(r.text,'html')
                from selenium import webdriver
            else:
                """not working yet due to Yahoo failure"""
                import requests
                from bs4 import BeautifulSoup
                url=('https://uk.finance.yahoo.com/quote/'+name)
                r=requests.get(url)
                web_content=BeautifulSoup(r.text,'html')
                print(web_content)
                web_content=web_content.find('span',{
                    'class':'D(ib) Fl(end) W(20%) Maw(300px) Cl(end)--mobxl W(250px)--tab768'
                })
                print(web_content)
    
    def MA(self,day=7):
        start_date=str(datetime.datetime.strptime(min(self.stock['Date'].values), "%Y-%m-%d")+ datetime.timedelta(days=day))
        end_date=max(self.stock['Date'].values)
        self.stock_MA=(((self.stock).rolling(window=day).mean()).loc[start_date:end_date, :])

    def derivative_rate(self,order=2):
        self.derivative_rate=[]
        df=self.stock.loc[:, self.stock.columns != 'Date']
        for i in range(1,order+1):
            n=df.shape[0]
            f_x=df.iloc[1:(n-1),].values
            f_a=df.iloc[0:(n-2),].values
            tmpr=f_x
            tmpr=tmpr-f_a
            tmpr=tmpr/f_a
            tmpr=pandas.DataFrame(tmpr)
            tmpr.columns=df.columns
            tmpr=tmpr.add_suffix('_'+str(i))
            tmpr.index=(self.stock.index)[0:(n-2)]
            self.derivative_rate.append(tmpr)
        print(self.derivative_rate)

    def plot(self):
        import plotly
        print(plotly.express.data.stocks())


    def prediction_RNN_black_box(self,data,days_ahead=4):
        X_train_list=[]
        X_train=[]
        i=0
        for stock in data:
            i+=1
            n=stock.stock.shape[0]
            X_train_list.append(stock.stock.iloc[0:(n-6),])
            for derivative in stock.derivative_rate:
                n=derivative.shape[0]
                print(derivative)
                X_train_list.append(derivative.iloc[0:(n-6),])
                tmpr=pandas.concat(X_train_list,axis=1)
            tmpr=tmpr.add_suffix('_'+str(i))
            X_train.append(tmpr)
        X_train=pandas.concat(X_train,axis=0)
        print(X_train)
        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(X_train)
        model=Sequential()
        model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(X_train,self.stock,epochs=25,batch_size=32)

if __name__=='__main__':
    Stock('RUI.PA')
    Stock('SAF.PA')
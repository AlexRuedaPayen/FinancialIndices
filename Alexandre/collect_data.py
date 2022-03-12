from pickle import FALSE
import pandas
import math
import datetime

"""from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM"""

class Stock:

    def __init__(self,name):
        self.stock=pandas.read_csv(filepath_or_buffer='../Data/'+name+'.csv')
        self.stock.index=self.stock['Date']
        print(f'Data from '+ str(min(self.stock['Date'].values))+' to '+str(max(self.stock['Date'].values)))
    
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
            
                
            
RUI_PA=Stock(name='RUI.PA')
VPK_AS=Stock(name='VPK.AS')
BP_L=Stock(name="BP.L")
SHELL_AS=Stock(name="SHELL.AS")
TTE_PA=Stock(name="TTE.PA")
XOM=Stock(name="XOM")

RUI_PA.MA()
VPK_AS.MA()
BP_L.MA()
SHELL_AS.MA()
TTE_PA.MA()
XOM.MA()

RUI_PA.derivative_rate()
VPK_AS.derivative_rate()
BP_L.derivative_rate()
SHELL_AS.derivative_rate()
TTE_PA.derivative_rate()
XOM.derivative_rate()


#RUI_PA.prediction_NN_black_box(data=[VPK_AS,BP_L,SHELL_AS,TTE_PA,XOM])

"""
BP_L.derivative_rate()
SHELL_AS.derivative_rate()
TTE_PA.derivative_rate()
XOM.derivative_rate()"""

import pandas_datareader

start=datetime.datetime(2020,1,1)
end=datetime.datetime(2022,1,1)

data=pandas_datareader.DataReader("MRNA","yahoo",start,end)
print(data)
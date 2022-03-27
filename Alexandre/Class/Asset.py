import os,sys
sys.path.append(os.getcwd())

import pandas
import math
import datetime
import numpy
from Alexandre.Class.Scrapper import Scrapper
from Alexandre.Class.Scrapper import Scrapper_financial_Yahoo,Scrapper_history_Yahoo,Scrapper_info_Yahoo
header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}

#import tensorflow

"""from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib.keras.models import Sequential,load_model
from tensorflow.contrib.keras.models import LSTM,Dense,Dropout"""

class Asset:

    def __init__(self,name,locally_stored=False):

        self.name=name

        if (locally_stored):
            self.history=pandas.read_csv(filepath_or_buffer='./Data/Class/Asset/history/'+name+'.csv')
            self.history.columns=['Date','Open','High','Low','Close','Close_Adj','Num_Transactions']
            self.history.index=self.history['Date']
            print(f'Data from '+ str(min(self.history['Date'].values))+' to '+str(max(self.history['Date'].values)))

        else:

            url_history=('https://uk.finance.yahoo.com/quote/'+self.name+'/history?p='+self.name)
            self.history=Scrapper_history_Yahoo(url=url_history)[0]
            self.history.columns=['Date','Open','High','Low','Close','Close_Adj','Num_Transactions']
            

    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.history.to_csv('./Data/Class/Asset/history/'+name+'.csv',header=True,encoding='utf-8',index=False)
    
    def MA(self,day=7):
        start_date=str(datetime.datetime.strptime(min(self.history['Date'].values), "%Y-%m-%d")+ datetime.timedelta(days=day))
        end_date=max(self.history['Date'].values)
        self.history_MA=(((self.history).rolling(window=day).mean()).loc[start_date:end_date, :])

    def derivative_rate(self,order=2):
        self.derivative_rate=[]
        df=self.history.loc[:, self.history.columns != 'Date']
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
            tmpr.index=(self.history.index)[0:(n-2)]
            self.derivative_rate.append(tmpr)

    def normalize_date(self,Assets):
        init_val_asset_denominator=self.history.iloc[self.history.shape[0]-1,1]
        self.assets_normize=dict()
        for key,asset in Assets.items():
            init_val_numerator=asset.history.iloc[asset.history.shape[0]-1,1]
            quot=init_val_asset_denominator/init_val_numerator
            tmpr=asset.history
            tmpr['Open']=[x*(quot) for x in asset.history['Open']]
            tmpr['Close']=[x*(quot) for x in asset.history['Close']]
            tmpr['High']=[x*(quot) for x in asset.history['High']]
            tmpr['Low']=[x*(quot) for x in asset.history['Low']]
            self.assets_normize[key]=tmpr
        

    def plot_price(self):

        import matplotlib.pyplot as plt

        self.history['Date'] = pandas.to_datetime(self.history['Date']).dt.date
        self.history=self.history.sort_values('Date')
        t=self.history['Date']
        
        Assets=self.assets_normize
        
        import colorsys
        N = len(Assets.keys())+1
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

        plt.plot_date(t,self.history['Open'],linestyle='solid',color=RGB_tuples[0])
        i=0

        name=[n for n,v in globals().items() if v == self][0]
        labels=[name]
        
        for key,value in Assets.items():
            i+=1
            value['Date'] = pandas.to_datetime(value['Date']).dt.date
            value=value.sort_values('Date')
            value.ffill(inplace=True)
            labels.append(key)
            plt.plot_date(t,value['Open'],linestyle='solid',color=RGB_tuples[i])
  
        plt.legend(labels, ncol=2)
        plt.savefig('./Data/Asset/fig/plot_'+name+'.png')


    def prediction_RNN_black_box(self,days_ahead=4):
        X_train_list=[]
        X_train=[]
        i=0
        for key,values in self.assets_normize.items():
            values.index=values["Date"].tolist()
            values=values.drop("Date",axis=1)
            values=values.add_suffix('_'+key)
            X_train_list.append(values)   
        X_train=pandas.concat(X_train_list,axis=1)
        print(X_train)
        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(X_train)
        print(scaled_data)
        model=Sequential()
        model.add(LSTM(units=50,return_sequences=True,input_shape=(scaled_data.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(X_train,self.history,epochs=25,batch_size=32)

if __name__=='__main__':

    Rubis=Asset('RUI.PA')
    Safran=Asset('SAF.PA')
    EDF=Asset('EDF.PA')
    Wheat=Asset('ZW%3DF')
    Brent=Asset('BZ%3DF')

    Rubis.normalize_date({'Safran':Safran,'EDF':EDF,'Wheat':Wheat,'Brent':Brent})
    Rubis.plot_price()

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
            self.history=Scrapper_history_Yahoo(url=url_history)
            self.history.columns=['Date','Open','High','Low','Close','Close_Adj','Num_Transactions']
            self.history.replace({'-':numpy.nan},inplace=True)
            self.history['Date']=pandas.to_datetime(self.history['Date'])
            self.history=self.history.astype(
                dtype={
                    'Open':'float32',
                    'High':'float32',
                    'Low':'float32',
                    'Close':'float32',
                    'Close_Adj':'float32',
                    'Num_Transactions':'float16' ###cannot handle NaN with integers
                }
            )
            self.assets_normize={}
            

    def disp(self):
        name=[n for n,v in globals().items() if v == self][0]
        print(
            "\n"+name+"\n"+
            "_______________"+"\n"
        )
        print(
            "History :"+"\n"
        )
        print(self.history)
        print(
            "\n"
        )
        print("_____________"+"\n")
        print(
            "Derivative :"+"\n"
        )
        for order,derivative_table in enumerate(self.derivative_rate):
            print("°°°°°°°°°°°°°°°°°°°")
            print("Derivate of order "+str(order+1)+" :")
            print(derivative_table)
            print("\n")

        self.plotter(df=self.history).show()

        for ind,value in enumerate(self.derivative_rate):
            self.plotter(df=value).show()

        return("\n")


    
    def __repr__(self):
        return(self.disp())

    def __str__(self):
        return(self.disp())

    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.history.to_csv('./Data/Class/Asset/history/'+name+'.csv',header=True,encoding='utf-8',index=False)
    
    def MA(self,day=7):
        self.history['Open_MA']=(self.history['Open']).rolling(window=day).mean()

    def derivative_rate_(self,order=2):
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
            df=tmpr
            tmpr=tmpr.assign(Date=(self.history.loc)[1:(n-2),'Date'].values)
            self.derivative_rate.append(tmpr)
        for ind,value in enumerate(self.derivative_rate):
            value.loc[:,value.columns != 'Date']*=100

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
        
    def plotter(self,df):
        import matplotlib.pyplot as plt

        df['Date'] = pandas.to_datetime(df['Date']).dt.date
        df=df.sort_values('Date')
        t=df['Date']
        
        Assets=self.assets_normize
        
        import colorsys
        N = len(Assets.keys())+1
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

        plt.plot_date(t,df['Open'],linestyle='solid',color=RGB_tuples[0])
        plt.plot_date(t,df['Open_MA'],linestyle='dashdot',color=RGB_tuples[0])
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
        return(plt)


    def correlation_coefficient(self,Asset,distribution):
        import scipy.stats
        Corr_dict=dict()
        for key,value in Asset.items():
            corr=0
            for ma,prob in distribution.items():
                self.MA(day=ma)
                value.MA(day=ma)
                self.derivative_rate_()
                value.derivative_rate_()

                a=self.derivative_rate[1].loc[:,['Date','Open_MA']]
                b=value.derivative_rate[1].loc[:,['Date','Open_MA']]

                a=a[~a.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]
                b=b[~b.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]
                print(a,b)

                n=pandas.concat(
                    objs=[a,b],
                    axis=1
                )

                print(n['Open_MA'])
                print(n['Open_MA'].iloc[:,0].values)
                print(n['Open_MA'].iloc[:,1].values)
                local=scipy.stats.pearsonr(n['Open_MA'].iloc[:,0].values,n['Open_MA'].iloc[:,1].values)
                print(local)
                corr+=prob*local
            Corr_dict[key]=corr
        return(Corr_dict)


    def prediction_price(self,method='PROPHET'):

        assert(method in set(['PROPHET','ARIMA','LTSM']))

        if method=='PROPHET':

            from fbprophet import Prophet

            fbp = Prophet(daily_seasonality = True)
            df=self.derivative_rate[1].loc[:,['Date','Open']]
            fbp.fit(df)
            fut = fbp.make_future_dataframe(periods=15) 
            forecast = fbp.predict(fut)

        if method=='ARIMA':

            from statsmodels.tsa.arima_model import ARIMA
            from sklearn.metrics import mean_squared_error

            df=self.derivative_rate[1].loc[:,['Date','Open']]
            train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
            training_data = train_data['Open'].values
            test_data = test_data['Open'].values   
            history = [x for x in training_data]
            model_predictions = []
            N_test_observations = len(test_data)
            for time_point in range(N_test_observations):
                model = ARIMA(history, order=(6,1,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                model_predictions.append(yhat)
                true_test_value = test_data[time_point]
                history.append(true_test_value)
            MSE_error = mean_squared_error(test_data, model_predictions)

        if method=='LTSM':
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.python.keras.models import Sequential,load_model
            from tensorflow.python.keras.models import LSTM,Dense,Dropout
            X_train_list=[]
            X_train=[]
            i=0
            for key,values in self.assets_normize.items():
                values.index=values["Date"].tolist()
                values=values.drop("Date",axis=1)
                #values=values.add_suffix('_'+key)
                X_train_list.append(values)
            print(X_train_list)   
            X_train=pandas.concat(X_train_list,axis=1)
            scaler=MinMaxScaler(feature_range=(0,1))
            scaled_data=scaler.fit_transform(X_train)
            model=Sequential()
            model.add(LSTM(units=50,return_sequences=True,input_shape=(scaled_data.shape[1],1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50,return_sequences=True))
            model.add(Dropout(0.2) )
            model.add(LSTM(units=50,return_sequences=True))
            model.add(Dense(units=1))
            model.compile(optimizer='adam',loss='mean_squared_error')
            model.fit(X_train,self.history,epochs=25,batch_size=32)


    def plot_price(self):
        name=[n for n,v in globals().items() if v == self][0]
        labels=[name]

        plt=self.plotter(self.history)
        plt.savefig('./Data/Asset/fig/plot_'+name+'.png')

if __name__=='__main__':

    Rubis=Asset('RUI.PA')
    Safran=Asset('SAF.PA')
    EDF=Asset('EDF.PA')
    """Wheat=Asset('ZW%3DF')
    Brent=Asset('BZ%3DF')"""

    Rubis.MA()
    Rubis.derivative_rate_()
    Rubis.normalize_date(Assets={'Safran':Safran,'EDF':EDF})
    Rubis.prediction_price(method='LTSM')
    #corr_Rubis=Rubis.correlation_coefficient({'Wheat':Wheat,'Brent':Brent},{i:1/7 for i in range(1,8)})

    #Rubis.normalize_date({'Safran':Safran,'EDF':EDF,'Wheat':Wheat,'Brent':Brent})
    #Rubis.plot_price()

import os,sys
sys.path.append(os.getcwd())

import pandas
import math
import datetime
import numpy
from Alexandre.Class.Scrapper import Scrapper
from Alexandre.Class.Scrapper import Scrapper_financial_Yahoo,Scrapper_history_Yahoo,Scrapper_info_Yahoo
header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}


class Asset:

    def __init__(self,name,locally_stored=False):

        self.name=name

        if os.path.isfile('./Data/Class/Asset/history/'+name+'.csv'):
            self.history=pandas.read_csv(filepath_or_buffer='./Data/Class/Asset/history/'+name+'.csv')
            self.history.columns=['Date','Open','High','Low','Close','Close_Adj','Num_Transactions']
            self.history.index=self.history['Date']
            print(f'Data already locally stored')
            print(f'Data from '+ str(min(self.history['Date'].values))+' to '+str(max(self.history['Date'].values)))

        else:
            print(f'Data not yet locally stored')
            print(f'Scrapping on Finance.Yahoo.com')
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
        self.history.to_csv('./Data/Class/Asset/history/'+self.name+'.csv',header=True,encoding='utf-8',index=False)
        print(f'File saved at '+'./Data/Class/Asset/history/'+self.name+'.csv')
 
    def derivative_rate_(self,order=2):
        self.derivative_rate=[]
        df=self.history.loc[:, self.history.columns != 'Date']
        for i in range(1,order+1):
            n=df.shape[0]
            f_x=df.iloc[1:(n-1),].values
            f_a=df.iloc[0:(n-2),].values
            tmpr=f_x
            tmpr=tmpr-f_a
            #tmpr=tmpr/f_a
            tmpr=pandas.DataFrame(tmpr)
            tmpr.columns=df.columns
            df=tmpr
            print(tmpr)
            print((self.history.loc)[1:(n-2),'Date'].values)
            tmpr=tmpr.assign(Date=(self.history.loc)[1:(n-2),'Date'].values)
            self.derivative_rate.append(tmpr)

    def normalize_date(self,Assets):
        init_val_asset_denominator=self.history.iloc[self.history.shape[0]-1,1]
        self.assets_normize=dict()
        for key,asset in Assets.items():
            init_val_numerator=asset.history.iloc[asset.history.shape[0]-1,1]
            quot=init_val_asset_denominator/init_val_numerator
            #tmpr=asset.history
            asset.history['Open']=[math.log(x*(quot)) for x in asset.history['Open']]
            asset.history['Close']=[math.log(x*(quot)) for x in asset.history['Close']]
            asset.history['High']=[math.log(x*(quot)) for x in asset.history['High']]
            asset.history['Low']=[math.log(x*(quot)) for x in asset.history['Low']]
            self.assets_normize[key]=asset
        self.history['Open']=[math.log(x) for x in self.history['Open'].values]
        self.history['Close']=[math.log(x) for x in self.history['Close'].values]
        self.history['High']=[math.log(x) for x in self.history['High'].values]
        self.history['Low']=[math.log(x) for x in self.history['Low'].values]
        
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


    def correlation_coefficient(self,distribution):
        import scipy.stats
        Corr_dict=dict()
        asset=self.assets_normize

   
        def MA(df,name='Open',day=7):
            df[name+'_MA']=(df[name]).rolling(window=day).mean()
            return(df)


        for key,value in asset.items():
            corr=0
            for ma,prob in distribution.items():
                self.derivative_rate_()
                value.derivative_rate_()

                a=self.derivative_rate[1].loc[:,['Date','Open']]
                b=value.derivative_rate[1].loc[:,['Date','Open']]

                print(a)

                a=MA(df=a,name='Open',day=ma)
                b=MA(df=b,name='Open',day=ma)

                a=a[~a.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]
                b=b[~b.isin([numpy.nan, numpy.inf, -numpy.inf]).any(1)]

                n=pandas.concat(
                    objs=[a,b],
                    axis=1
                )

                print(n)
                local=scipy.stats.pearsonr(n['Open_MA'].iloc[:,0].values,n['Open_MA'].iloc[:,1].values)
                print(local)
                corr+=prob*local[1]
            Corr_dict[key]=corr

        name=[n for n,v in globals().items() if v == self][0]

        for key,value in Corr_dict.items():
            print("Correlation "+name+"/"+key+" : {:.1f}%".format(value*100))


    def prediction_price(self,method='ARIMA'):

        assert(method.issubset(set(['ARIMA','LTSM'])))

        list_df=dict()
        self.history.sort_values(by='Date', axis=0, ascending=True, inplace=True)

        if 'ARIMA' in method:

            from statsmodels.tsa.arima.model import ARIMA
            from sklearn.metrics import mean_squared_error

            df=self.history.loc[:,['Date','Open']]

            train_data, test_data = df[0:int((df.shape[0])*0.7)], df[int((df.shape[0])*0.7):]
            training_data = train_data['Open'].values
            test_data = test_data['Open'].values   
            history = [x for x in training_data]
            model_predictions = []
            N_test_observations = len(test_data)
            for time_point in range(N_test_observations):
                model = ARIMA(history, order=(6,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                model_predictions.append(yhat)
                true_test_value = test_data[time_point]
                history.append(true_test_value)
            MSE_error = mean_squared_error(test_data, model_predictions)
            print("MSE error : {:f}".format(MSE_error))

            print(100-(df.shape[0])*0.7)
            print(len(model_predictions))
            df['Open_prediction'] = [math.exp(x) for x in df['Open'].values]
            df.loc[int((df.shape[0])*0.7):,'Open_prediction'] = [math.exp(x) for x in model_predictions]

            list_df['ARIMA']=df
            

        if 'LTSM' in method:

            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.python.keras.engine.sequential import Sequential
            from tensorflow.python.keras.layers import LSTM,Dense,Dropout,Input
            
            values=self.history[['Open']]
            values.index=self.history["Date"].tolist()
            X_train_list=[values]
            i=0
            for key,val in self.assets_normize.items():
                values=val.history
                values.sort_values(by='Date', axis=0, ascending=True, inplace=True)
                values.index=values["Date"].tolist()
                values=values[['Open']]
                values=values.add_suffix("_"+key)
                X_train_list.append(values) 
            Y_train=X_train_list[0]
            X_train=pandas.concat(X_train_list[1:],axis=1)

            X_train,X_test=X_train[0:int((X_train.shape[0])*0.7)], X_train[int((X_train.shape[0])*0.7):]
            Y_train,Y_test=Y_train[0:int((Y_train.shape[0])*0.7)], Y_train[int((Y_train.shape[0])*0.7):]

            scaled_x_train = X_train.values.tolist()
            scaled_x_train = numpy.asarray(scaled_x_train).astype('float32')

            scaled_x_test = X_test.values.tolist()
            scaled_x_test = numpy.asarray(scaled_x_test).astype('float32')
           
            model=Sequential()
            model.add(LSTM(units=50,return_sequences=True,input_shape=(scaled_x_train.shape[1],1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50,return_sequences=True))
            model.add(Dropout(0.2) )
            model.add(LSTM(units=50,return_sequences=False))
            model.add(Dense(units=1))
            model.compile(optimizer='adam',loss='mean_squared_error')
            scaled_y_train=numpy.asarray(Y_train.values.tolist()).astype('float32')

            scaled_x_train=numpy.reshape(scaled_x_train,(scaled_x_train.shape[0],scaled_x_train.shape[1],1))
            scaled_y_train=numpy.reshape(scaled_y_train,(scaled_y_train.shape[0],scaled_y_train.shape[1],1))

            scaled_x_test=numpy.reshape(scaled_x_test,(scaled_x_test.shape[0],scaled_x_test.shape[1],1))
            model.fit(scaled_x_train,scaled_y_train,epochs=25,batch_size=32)


            scaled_train_prediction=model.predict(scaled_x_train)
            scaled_test_prediction=model.predict(scaled_x_test)
            train_prediction=[math.exp(x) for x in scaled_train_prediction]
            test_prediction=[math.exp(x) for x in scaled_test_prediction]

            prediction=train_prediction+test_prediction

            print(len(prediction))
       
            Y=[math.exp(x) for x in self.history['Open'].values]

            df=pandas.DataFrame({'Date':self.history["Date"].tolist(),'Open':Y,'Open_prediction':prediction})
            list_df['LTSM']=df

        import matplotlib.pyplot as pyplot

        name=[n for n,v in globals().items() if v == self][0]

        fig,ax=pyplot.subplots(figsize=(8,4))

        import colorsys
        N = len(list_df.keys())+1
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
        i=0

        t=df['Date']
        ax.plot_date(t,df['Open'],color=RGB_tuples[i],linestyle='solid',label='Price of stock '+name)
        for key,value in list_df.items():
            i+=1
            t=value['Date']
            pyplot.plot_date(t,value['Open_prediction'],linestyle='solid',color=RGB_tuples[i],label='Predicted price of '+name+' with '+key)
        pyplot.legend()
        #print(os.getcwd())
        pyplot.savefig(os.getcwd()+'/Data/Class/Asset/fig/plot_prediction_price_'+name+'.png')
        pyplot.show()

        return(list_df)
            

if __name__=='__main__':

    Rubis=Asset('RUI.PA')
    Safran=Asset('SAF.PA')
    EDF=Asset('EDF.PA')

    Rubis.save()
    Safran.save()
    EDF.save()
    """Wheat=Asset('ZW%3DF')
    Brent=Asset('BZ%3DF')"""

    Rubis.normalize_date(Assets={'Safran':Safran,'EDF':EDF})
    corr_Rubis=Rubis.correlation_coefficient({i:1/7 for i in range(1,8)})
    Rubis.prediction_price(method=set(['LTSM','ARIMA']))

    #Rubis.normalize_date({'Safran':Safran,'EDF':EDF,'Wheat':Wheat,'Brent':Brent})
    #Rubis.plot_price()

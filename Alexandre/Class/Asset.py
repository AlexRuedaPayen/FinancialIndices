import pandas
import math
import datetime
import numpy
import Scrapper
header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}

#import tensorflow

"""from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout"""

class Asset:

    def __init__(self,name,locally_stored=False):

        self.name=name

        if (locally_stored):
            self.stock=pandas.read_csv(filepath_or_buffer='./Data/'+name+'.csv')
            self.stock.index=self.stock['Date']
            print(f'Data from '+ str(min(self.stock['Date'].values))+' to '+str(max(self.stock['Date'].values)))

        else:
            
            scheme={
                'tr':{
                    'class_':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)',
                    'row':{
                        'td':{'class_':'Py(10px)'}
                    }
                }
            }

            url=('https://uk.finance.yahoo.com/quote/'+self.name+'/history?p='+self.name)
            Scrapper_Press_Releases=Scrapper(scheme=scheme,header=header,type='table')
            self.financial=Scrapper_Press_Releases(url=url)
        


    def scrap_infos(self,website='FMP'):

        if website=='FMP':

            url_news=('https://site.financialmodelingprep.com/historical-data/'+self.name)
            scheme={
                    'a': {
                        'class_': 'article-item', 
                        'row': {
                            'h5': {'class_': 'article-date'},
                            'h4': {'class_': 'article-title'},
                            'p': {'class_': 'article-text'}
                        }
                }
            }
            Scrapper_Press_Releases=Scrapper(scheme=scheme,header=header,type='table')
            self.financial=Scrapper_Press_Releases(url=url_news)
            


        if website=='Yahoo':

            url_news=('https://finance.yahoo.com/quote/'+self.name+'/news?p='+self.name)
            url_press_releases=('https://finance.yahoo.com/quote/'+self.name+'/press-releases?p='+self.name)
            url_financials=('https://finance.yahoo.com/quote/'+self.name+'/financials?p='+self.name)


            scheme={
                'div':{
                    'class_':'D(tbr)',
                    'row':{
                        'div':'D(tbc)'
                    }
                }
            }
            Scrapper_Press_Releases=Scrapper(scheme=scheme,header=header,type='table')
            self.news=Scrapper_Press_Releases(url=url_news)
            self.press_releases=Scrapper_Press_Releases(url=url_press_releases)
            self.financial=Scrapper_Press_Releases(url=url_financials)




    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.stock.to_csv('./Data/Asset/'+name+'.stock..csv',header=True,encoding='utf-8',index=False)
        self.news.to_csv('./Data/Asset/'+name+'.news.csv',header=True,encoding='utf-8',index=False)
        self.press_releases.to_csv('./Data/Asset/'+name+'.press_releases.csv',header=True,encoding='utf-8',index=False)
        self.financial.to_csv('./Data/Asset/'+name+'.financial.csv',header=True,encoding='utf-8',index=False)
    
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

    def normalize_date(self,Assets):
        init_val_asset_denominator=self.stock.iloc[self.stock.shape[0]-1,1]
        self.assets_normize=dict()
        for key,asset in Assets.items():
            init_val_numerator=asset.stock.iloc[asset.stock.shape[0]-1,1]
            quot=init_val_asset_denominator/init_val_numerator
            tmpr=asset.stock
            tmpr['Open']=[x*(quot) for x in asset.stock['Open']]
            tmpr['Close']=[x*(quot) for x in asset.stock['Close']]
            tmpr['High']=[x*(quot) for x in asset.stock['High']]
            tmpr['Low']=[x*(quot) for x in asset.stock['Low']]
            self.assets_normize[key]=tmpr
        

    def plot_price(self):

        import matplotlib.pyplot as plt
        import datetime
        import re

        self.stock['Date'] = pandas.to_datetime(self.stock['Date']).dt.date
        self.stock=self.stock.sort_values('Date')
        t=self.stock['Date']
        
        Assets=self.assets_normize
        
        import colorsys
        N = len(Assets.keys())+1
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

        plt.plot_date(t,self.stock['Open'],linestyle='solid',color=RGB_tuples[0])
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
        model.fit(X_train,self.stock,epochs=25,batch_size=32)

if __name__=='__main__':

    Rubis=Asset('RUI.PA')
    Rubis.scrap_infos()

    Safran=Asset('SAF.PA')
    Safran.scrap_infos()

    EDF=Asset('EDF.PA')
    EDF.scrap_infos()

    Wheat=Asset('ZW%3DF')
    Wheat.scrap_infos()

    Brent=Asset('BZ%3DF')
    Brent.scrap_infos()


    print(Safran.stock)
    print(EDF.stock)

    print(Rubis.press_releases)
    print(Safran.press_releases)
    print(EDF.press_releases)
    #print(Wheat.press_releases)
    #print(Brent.press_releases)

    """print(Rubis.news)
    print(Safran.news)
    print(EDF.news)
    print(Wheat.news)
    print(Brent.news)"""
    
    print(Rubis.financial)
    print(Safran.financial)
    print(EDF.financial)

    """Safran=Asset('SAF.PA')
   

    Rubis.derivative_rate()
    Safran.derivative_rate()
    Wheat.derivative_rate()
    Brent.derivative_rate()

    Rubis.save()
    Safran.save()
    Wheat.save()
    Brent.save()"""

    Rubis.normalize_date({
        'Safran':Safran,
        'Wheat':Wheat,
        'Brent':Brent
    })
    
    Rubis.plot_price()
    #Rubis.prediction_RNN_black_box()
    #Rubis.shapley_value()
import pandas
import math
import datetime
import numpy

class Asset:

    def __init__(self,name,locally_stored=False):

        self.name=name

        if (locally_stored):
            self.stock=pandas.read_csv(filepath_or_buffer='./Data/'+name+'.csv')
            self.stock.index=self.stock['Date']
            print(f'Data from '+ str(min(self.stock['Date'].values))+' to '+str(max(self.stock['Date'].values)))

        else:
            datelist=[] 
            lowlist=[]
            highlist=[]
            openlist=[]
            closelist=[]

            import requests
            from bs4 import BeautifulSoup
            from selenium import webdriver

            url=('https://uk.finance.yahoo.com/quote/'+self.name+'/history?p='+self.name)
            r=requests.get(url,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
            web_content=BeautifulSoup(r.text,'html')
            
            web_content=web_content.find('div',{'class':'Pb(10px) Ovx(a) W(100%)'})
            web_content=web_content.find_all('tr',{'class':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'})

            for x in web_content:
                datelist.append(x.find('td',{'class':'Py(10px) Ta(start) Pend(10px)'}).text)
                val=x.find_all('td',{'class':'Py(10px) Pstart(10px)'})
                openlist.append(val[0].text)
                highlist.append(val[1].text)
                lowlist.append(val[2].text)
                closelist.append(val[3].text)

            def float_(x):
                try:
                    return(float(x.replace(",","")))
                except:
                    return(numpy.nan)

            self.stock=pandas.DataFrame({
                'Date':[x for x in datelist],
                'Open':[float_(x) for x in openlist],
                'Close':[float_(x) for x in closelist],
                'High':[float_(x) for x in highlist],
                'Low':[float_(x) for x in lowlist]
            })

    def work_onlive(self,duration=360,interval=6):
        import requests
        from bs4 import BeautifulSoup

        url=('https://uk.finance.yahoo.com/quote/'+self.name+'/history?p='+self.name)
        r=requests.get(url,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
        
        web_content=BeautifulSoup(r.text,'html')
        web_content=web_content.find('div',{'class':'D(ib) Mend(20px)'})
        web_content=web_content.find('span',{'class':'_11248a25 _8e5a1db9'})
        print(web_content)

    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.stock.to_csv('./Data/Asset/'+name+'.csv',header=True,encoding='utf-8',index=False)
    
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
        
        for key,value in Assets.items():
            i+=1
            value['Date'] = pandas.to_datetime(value['Date']).dt.date
            value=value.sort_values('Date')
            value.ffill(inplace=True)
         
            plt.plot_date(t,value['Open'],linestyle='solid',color=RGB_tuples[i])

        plt.show()

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
    Rubis=Asset('RUI.PA')
    Safran=Asset('SAF.PA')
    Wheat=Asset('ZW%3DF')
    Brent=Asset('BZ%3DF')

    print(Rubis.stock)
    print(Safran.stock)
    print(Wheat.stock)
    print(Brent.stock)

    Rubis.derivative_rate()
    Safran.derivative_rate()
    Wheat.derivative_rate()
    Brent.derivative_rate()

    print(Rubis.derivative_rate)
    print(Safran.derivative_rate)
    print(Wheat.derivative_rate)
    print(Brent.derivative_rate)

    Rubis.save()
    Safran.save()
    Wheat.save()
    Brent.save()

    Rubis.normalize_date({
        'Safran':Safran,
        'Wheat':Wheat,
        'Brent':Brent
    })
    Rubis.plot_price()

    print(a)
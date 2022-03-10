import pandas
import numpy
import datetime

class Stock:
    def __init__(self,name):
        self.stock=pandas.read_csv(filepath_or_buffer='../Data/'+name+'.csv')
        self.stock.index=self.stock['Date']
        print(f'Data from '+ str(min(self.stock['Date'].values))+' to '+str(max(self.stock['Date'].values)))
    def MA(self,day=7):
        start_date=str(datetime.datetime.strptime(min(self.stock['Date'].values), "%Y-%m-%d")+ datetime.timedelta(days=day))
        end_date=max(self.stock['Date'].values)
        self.stock_MA=(((self.stock).rolling(window=day).mean()).loc[start_date:end_date, :])

RUI_PA=Stock(name='RUI.PA')
RUI_PA.MA()

print(RUI_PA.stock)
print(RUI_PA.stock_MA)

import os,sys
sys.path.append(os.getcwd())

from Alexandre.Class.Asset import *

class Stock(Asset):

    def __init__(self,name,locally_stored=False):

        Asset.__init__(name,locally_stored)

        if locally_stored:

            self.news=pandas.read_csv(filepath_or_buffer='./Data/Class/Stock/news/'+name+'.csv')
            self.press_releases=pandas.read_csv(filepath_or_buffer='./Data/Class/Stock/press-releases/'+name+'.csv')
            self.financials=pandas.read_csv(filepath_or_buffer='./Data/Class/Stock/financials/'+name+'.csv')

        else:

            url_news=('https://finance.yahoo.com/quote/'+self.name+'/news?p='+self.name)
            url_press_releases=('https://finance.yahoo.com/quote/'+self.name+'/press-releases?p='+self.name)
            url_financials=('https://finance.yahoo.com/quote/'+self.name+'/financials?p='+self.name)


            self.financials=Scrapper_financial_Yahoo(url=url_financials)
            self.press_releases=Scrapper_info_Yahoo(url=url_press_releases)
            self.news=Scrapper_info_Yahoo(url=url_news)

    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.news.to_csv('./Data/Class/Stock/news/'+name+'.csv',header=True,encoding='utf-8',index=False)
        self.press_releases.to_csv('./Data/Class/Stock/press_releases/'+name+'.csv',header=True,encoding='utf-8',index=False)
        self.financial.to_csv('./Data/Class/Stock/financials/'+name+'.csv',header=True,encoding='utf-8',index=False)

if __name__=='__main__':

    Rubis=Stock('RUI.PA')
    Safran=Stock('SAF.PA')
    EDF=Stock('EDF.PA')

    Rubis.save()
    Safran.save()
    EDF.save()
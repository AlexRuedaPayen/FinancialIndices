import pandas
import requests
from bs4 import BeautifulSoup
from selenium import webdriver


class Scrapper:

    def __init__(self,scheme,header,type):

        assert(type(header) is dict)
        assert(type(scheme) is dict)
        assert(list(scheme.keys()) in set(['a','div','p']))
        assert(type(scheme.values()) is dict or type(scheme.values) is str)

        for i in scheme.values():
            if (i is dict):
                assert(type(i.keys()) in set(['class_,row']))
                assert(type(i['class_']) is str)
                assert(type(i['row']) is dict)
                assert(i['row'].keys() in set(['h1','h2','h3','h4','h5']))
                assert(type(i['row'].values()) is dict)
                for j in i['row'].values():
                    assert(j.keys() in set(['class_']))
                    assert(type(j.values()) is str)
    
        self.scheme=scheme
        self.header=header
        self.type=type

    def fill_table(self,url,header=True):
        scheme=self.scheme
        r=requests.get(url,headers=self.header)
        web_content=BeautifulSoup(r.text,'html')
        df=[]
        for key1,value1 in scheme.items():
            web_content_key1=web_content.find_all(key1, class_=value1['class_'])
            df_list=[]
            row_=[]
            for row in web_content_key1[header:]:
                for key2,value2 in value1['row'].items():
                    x_val=row.find_all(key2, class_=value2['class_'])
                    for y in x_val:
                        y_text=(''.join([z.replace(",","") for z in y.text]))
                        row_.append(y_text)
                    df_list.append(row_)
                    row_=[]
            df_tmpr=pandas.DataFrame(df_list)
            df.append(df_tmpr)
        return(df)    

    def fill_boxes(self,url):
            r=requests.get(url,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
            web_content=BeautifulSoup(r.text,'html')
            df=[]
            row_=[]
            for key1,value1 in scheme.items():
                web_content_key1=web_content.find_all(key1, class_=value1['class_'])
                df_list=[]
                for row in web_content_key1:
                    for key2,value2 in value1['row'].items():
                        tmpr=row.find(key2,value2['class_'])
                        if (tmpr==None):
                            row_.append('')
                            continue
                        row_.append(row.find(key2,value2['class_']).text)
                    df_list.append(row_)
                    row_=[]
                    print("________")
                df_tmpr=pandas.DataFrame(df_list)
                df.append(df_tmpr)

    def __call__(self,url):
        if (self.type=='box'):
            return(self.fill_boxes(url))
        if (self.type=='table'):
            return(self.fill_table(url))

if __name__=='__main__':

    header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}

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

    url='https://site.financialmodelingprep.com/historical-data/'+'EDF.PA'

    Scrapper_EDF=Scrapper(scheme=scheme,header=header)
    print(Scrapper_EDF(tye='table',url=url))

    scheme={
                'div':{
                    'class_':'D(tbr)',
                    'row':{
                        'div':{'class_':'D(tbc)'}
                    }
                }
            }
    
    url='https://finance.yahoo.com/quote/EDF.PA/financials?p=EDF.PA'

    Scrapper_Financials=Scrapper(scheme=scheme,header=header)
    print(Scrapper_Financials(type='table',url=url))

    scheme={
        'tr':{
            'class_':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)',
            'row':{
                'td':{'class_':'Py(10px)'}
            }
        }
    }

    url=('https://uk.finance.yahoo.com/quote/EDF.PA/history?p=EDF.PA')

    Scrapper_History=Scrapper(scheme=scheme,header=header)
    print(Scrapper_History(type='table',url=url))

    url=(('https://finance.yahoo.com/quote/EDF.PA/press-releases?p=EDF.PA'))

    scheme={
        'div':{
            'class_':'Cf',
            'row':{
                'div':{'class_':'C(#959595) Fz(11px) D(ib) Mb(6px)'},
                'h3':{'class_':'Mb(5px)'},
                'p':{'class_':'Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(3,57px) LineClamp(3,51px)--sm1024 M(0)'}
            }
        }
    }

    Scrapper_Press_Releases=Scrapper(scheme=scheme,header=header)
    print(Scrapper_Press_Releases(type='box',url=url)
import pandas
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}


class Scrapper:

    def __init__(self,scheme,header,type):

        """assert(type(header) is dict)
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
                    assert(type(j.values()) is str)"""
    
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
            colname=[]
            for row in web_content_key1[header:]:
                for key2,value2 in value1['row'].items():
                    x_val=row.find_all(key2, class_=value2['class_'])
                    for y in x_val:
                        y_text=(''.join([z.replace(",","") for z in y.text]))
                        row_.append(y_text)
                    df_list.append(row_)
                    row_=[]
            df_tmpr=pandas.DataFrame(df_list)
            if header:
                header_=list(value1['header'].keys())[0]
                class__=value1['header'][header_]['class_']
                print(header_,class__)
                for name in web_content_key1[0].find_all(header_,class_=class__):
                    colname.append(name.text)
                df_tmpr.columns=colname
            else :
                header_=list(value1['header'].keys())
                print(header_)
                class__=[(h,v) for h in header_ for k,v in value1['header'][h].items() ]
                print(class__)
                for h,v in class__:
                    name=row.find_all(h,class_=v)
                    colname=colname+[n.text for n in name]
                if (len(colname)==df_tmpr.shape[1]):
                    df_tmpr.columns=colname
                else:
                    print('Error entering header parameters')
            df.append(df_tmpr)
        return(df)    

    def fill_boxes(self,url):
        r=requests.get(url,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
        web_content=BeautifulSoup(r.text,'html')
        df=[]
        for key1,value1 in scheme.items():
            web_content_key1=web_content.find_all(key1, class_=value1['class_'])
            df_list=[]
            if ('row' in value1.keys()):
                row_=dict()
                for row in web_content_key1:
                    for key2,value2 in value1['row'].items():
                        tmpr=row.find(key2,value2['class_'])
                        if not value2['name'] in row_.items():
                            row_[value2['name']]=list()
                        if (tmpr==None):
                            row_[value2['name']].append('')
                            continue
                        row_[value2['name']].append(row.find(key2,value2['class_']).text)
                    df_list.append(row_)
            else:
                row_={value1['name'] : list()}
                for row in web_content_key1:
                    row_[value1['name']].append(row.text)
                df_list.append(row_)
            df_tmpr=pandas.DataFrame(data=row_)
            df.append(df_tmpr)
        return(df)   


    def __call__(self,url,header=False):
        if (self.type=='box'):
            return(self.fill_boxes(url,header))
        if (self.type=='table'):
            return(self.fill_table(url,header))



scheme_financial_Yahoo={
    'div':{
        'header':{
            'div':{'class_':'D(ib)'}
        },
        'class_':'D(tbr)',
        'row':{
            'div':{'class_':'D(tbc)'}
        }
    }
}
scheme_history_Yahoo={
    'tr':{
        'class_':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)',
        'row':{
            'td':{'class_':'Py(10px)'}
        }
    }
}   
scheme_info_Yahoo={
    'div':{
        'class_':'Cf',
        'row':{
            'div':{'class_':'C(#959595) Fz(11px) D(ib) Mb(6px)'},
            'h3':{'class_':'Mb(5px)'},
            'p':{'class_':'Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(3,57px) LineClamp(3,51px)--sm1024 M(0)'}
        }
    }
}

Scrapper_financial_Yahoo=Scrapper(scheme=scheme_financial_Yahoo,header=header,type='table')
Scrapper_history_Yahoo=Scrapper(scheme=scheme_history_Yahoo,header=header,type='table')
Scrapper_info_Yahoo=Scrapper(scheme=scheme_info_Yahoo,header=header,type='boxes')

 


if __name__=='__main__':

    scheme={
                'div':{
                    'header':{
                        'div':{'class_':'D(ib)'}
                    },
                    'class_':'D(tbr)',
                    'row':{
                        'div':{'class_':'D(tbc)'}
                    }
                }
            }
    
    url='https://finance.yahoo.com/quote/EDF.PA/financials?p=EDF.PA'

    Scrapper_Financials=Scrapper(scheme=scheme,header=header,type='table')
    print(Scrapper_Financials(url=url,header=True))

    print("___________-")
    scheme={
        'tr':{
            'header':{
                'th':{
                    'class1':'C($tertiaryColor) Fz(xs) Ta(end)',
                    'class2':'Fw(400) Py(6px)'
                }
            },
            'class_':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)',
            'row':{
                'td':{'class_':'Py(10px)'}
            }
        }
    }

    url=('https://uk.finance.yahoo.com/quote/EDF.PA/history?p=EDF.PA')

    Scrapper_History=Scrapper(scheme=scheme,header=header,type='table')
    print(Scrapper_History(url=url,header=False))

    print(False)

    url=(('https://finance.yahoo.com/'))
    scheme={
        'div':{
            'class_':'Pos(a) B(0) Start(0) End(0) Bg($ntkLeadGradient) Pstart(25px) Pstart(18px)--md1100 Pt(50px) Pend(45px) Pend(25px)--md1100 Bdrsbend(2px) Bdrsbstart(2px)',
            'row':{
                'h2':{
                    'class_':'Fz(22px)--md1100 Lh(25px)--md1100 Fw(b) Tsh($ntkTextShadow) Td(u):h Fz(25px) Lh(31px)',
                     'name':'headline'
                },
                'p':{
                    'class_':'Fz(12px) Fw(n) Lh(14px) LineClamp(3,42px) Pt(6px) Tov(e)',
                    'name':'summary'
                }
            }
        },
        'h3':{
            'class_':'Fz(14px)--md1100 Lh(16px)--md1100 Fw(700) Fz(16px) Lh(18px) LineClamp(3,54px) Va(m) Tov(e)',
            'name':'headline'
        },
        'div':{
            'class_':'Ov(h) Pend(44px) Pstart(25px)',
            'row':{
                'div':{
                    'class_':'Fz(12px) Fw(b) Tt(c) D(ib) Mb(6px) C($c-fuji-blue-1-a) Mend(9px) Mt(-2px)',
                    'name':'type'
                },
                'h3':{
                    'class_':'Mb(5px)',
                    'name':'headline'
                },
                'p':{
                    'class_':'Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0)',
                    'name':'subheadline'
                },
                'div':{
                    'class_':'Lh(15px) Fw(b) LineClamp(3,45px) D(i)--sm1024 Pend(10px)--sm1024',
                    'name':'summary'
                }
            }
        }
    }
    Scrapper_Press_Releases=Scrapper(type='box',scheme=scheme,header=header)
    print((Scrapper_Press_Releases(url=url)))


    """
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

    url=(('https://www.bloomberg.com/europe'))
    scheme={
        'section':{
            'class_':'single-story-module__eyebrow'
        },
        'div':{
            'class_':'single-story-module__related-story-eyebrow'
        },
        'div':{
            'class_':'story-package-module__stories',
            'div':{
                'class_':'div',
                'row':{
                    'h3':{'class':'story-package-module__story__headline'},
                    'div':{'div':'story-package-module__story__summary'}
                }
            }
        }
    }
    Scrapper_Press_Releases=Scrapper(type='box',scheme=scheme,header=header)
    print(Scrapper_Press_Releases(url=url))

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
    print(Scrapper_EDF(tye='table',url=url))"""

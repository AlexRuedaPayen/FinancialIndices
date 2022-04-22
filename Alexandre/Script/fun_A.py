import os,sys,json
from ssl import AlertDescription
from trace import CoverageResults
import pandas
print(os.getcwd())
sys.path.append(os.getcwd())


#device_path="/home/MacAlexandre_GCP_VM1/Projects/"
#sys.path.append(device_path+"Financial_Indices/")

def fun_A():

    from Alexandre.Class.Reddit import Reddit
    from Alexandre.Class.Scrapper import Scrapper


    header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
    url_Ukraine_cities=('https://en.wikipedia.org/wiki/Control_of_cities_during_the_Russo-Ukrainian_War')
    Scrapper_Wikipedia=Scrapper(scheme={},header=header,type='wikipedia_table')
    GEO_Ukraine=Scrapper_Wikipedia(url=url_Ukraine_cities)
    GEO_Ukraine.columns=['City','Population','Krai','Country']

    Ukraine=Reddit(topic="Ukraine",
                   top=100,
                   attributes=['headlines','id','author','created_utc','score','upvote_ratio','url'])
    Ukraine.topic_model_LDA()
    Ukraine.sentiment_analysis()
    Ukraine.Map_Domains()

    Ukraine.redit_data=Ukraine.redit_data.astype({'headlines': 'str','id': 'str','author': 'str','url':'str'})
    print(Ukraine.redit_data.dtypes)

    GEO=pandas.read_csv(filepath_or_buffer='./Data/Miscallaneous/Geography/worldcities.csv',sep=",",header=1)
    print({key:value for key,value in Ukraine.domains_inv.items() if key in set(GEO_Ukraine['City'].values)})

    JSON={}
    JSON['reddit_data']=Ukraine.redit_data.to_dict()
    JSON['topic_list']={x[0]:x[1] for x in Ukraine.topic_list}

    """with open("./Data/Script/fun_A.json","w") as f:
        f.write(json.dumps(JSON,indent=4))"""


if __name__=='__main__':
    fun_A()
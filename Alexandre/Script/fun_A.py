import os,sys,json
from ssl import AlertDescription
from trace import CoverageResults
#print(os.getcwd())
#sys.path.append(os.getcwd())


device_path="/home/MacAlexandre_GCP_VM1/Projects/"
sys.path.append(device_path+"Financial_Indices/")

def fun_A():
    from Alexandre.Class.Reddit import Reddit

    Ukraine=Reddit(topic="Ukraine",
                   top=100,
                   attributes=['headlines','id','author','created_utc','score','upvote_ratio','url'])
    Ukraine.topic_model_LDA()
    Ukraine.sentiment_analysis()

    Ukraine.redit_data=Ukraine.redit_data.astype({'headlines': 'str','id': 'str','author': 'str','url':'str'})
    print(Ukraine.redit_data.dtypes)

    JSON={}
    JSON['reddit_data']=Ukraine.redit_data.to_dict()
    JSON['topic_list']={x[0]:x[1] for x in Ukraine.topic_list}

    with open("./Data/Script/fun_A.json","w") as f:
        f.write(json.dumps(JSON,indent=4))


if __name__=='__main__':
    fun_A()
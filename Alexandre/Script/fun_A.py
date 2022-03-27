import os,sys,json
from ssl import AlertDescription
#print(os.getcwd())
#sys.path.append(os.getcwd())



device_path="/home/MacAlexandre_GCP_VM1/Projects/"
sys.path.append(device_path+"Financial_Indices/")

def fun_A():
    """from Alexandre.Class.Reddit import Reddit

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
        f.write(json.dumps(JSON,indent=4))"""

    JSON={}
    JSON['reddit_data']={'col1':[1,2,3],'col2':[2,3,4]}
    JSON['topic_list']={'col1':[0],'col2':['0.5*cat +0.5*dog']}
    with open(device_path+"Financial_Indices/Data/Script/fun_A.json","w") as f:
        f.write(json.dumps(JSON,indent=4))
    print(device_path+"Financial_Indices/Data/Script/fun_A.json created with success")

if __name__=='__main__':
    fun_A()
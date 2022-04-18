import os,sys,json
from ssl import AlertDescription
from trace import CoverageResults
#print(os.getcwd())
#sys.path.append(os.getcwd())


device_path="/home/MacAlexandre_GCP_VM1/Projects/"
sys.path.append(device_path+"Financial_Indices/")

def fun_Asset():
    from Alexandre.Class.Asset import Asset 

    Rubis=Asset('RUI.PA')
    Safran=Asset('SAF.PA')
    EDF=Asset('EDF.PA')

    Rubis.normalize_date(Assets={'Safran':Safran,'EDF':EDF})
    corr_Rubis=Rubis.correlation_coefficient({i:1/7 for i in range(1,8)})
    list_df=Rubis.prediction_price(method=set(['LTSM']))

    JSON={}
    JSON['history']=Rubis.history.to_dict()
    JSON['corr_Rubis']=corr_Rubis
    JSON['prediction_price']=list_df

    with open("./Data/Script/fun_Asset.json","w") as f:
        f.write(json.dumps(JSON,indent=4))


if __name__=='__main__':
    fun_Asset()
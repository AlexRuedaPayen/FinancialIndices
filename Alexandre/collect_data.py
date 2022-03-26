from pickle import FALSE
import pandas
import math
import datetime
import praw
import os,paramiko,subprocess

"""from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM"""


class GCP:
    def __init__(self,host="MacAlexandre@34.125.182.253",
                 keyfile="~/.ssh/VM-1-GCP-Instance1/key",
                 class_=["history"],
                 data_=["RUI.PA"]):
        ssh = paramiko.SSHClient()
        self.ssh=ssh 
        k = paramiko.RSAKey.from_private_key_file(keyfile)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username="AlexRuedaPayen", pkey=k)
        self.class_=class_

    def __enter__(self):
        file_path_origin="./Alexandre/class/"
        file_path_destination="~/Projects/Financial_Indices/Alexandre/class/"
        for i in self.class_:
            file_path_class_origin=file_path_origin+i
            file_path_class_destination=file_path_destination+i
            subprocess.run(["scp", file_path_class_origin, "USER@SERVER:"+file_path_class_destination])
        return()

    def run(self,script):
        file_path_script_origin="./Alexandre/script/"+script
        file_path_script_destination="~/Projects/Financial_Indices/Alexandre/script/"+script
        subprocess.run(["scp", file_path_script_origin, "USER@SERVER:"+file_path_script_destination])
        subprocess.run(["python3", file_path_script_destination])


    def __exit__(self):
        for i in file:
            subprocess.run(["scp", "USER@SERVER:"+i,i])
        self.ssh.disconnect()

def funA():
    pass

def funB():
    pass

def funC():
    pass

with GCP(host="MacAlexandre@34.125.182.253",
        keyfile="~/.ssh/VM-1-GCP-Instance1/key",
        class_=["history"],
        data_=["RUI.PA"]) as f:
        f.run(script='funA.py')
        f.run(script='funB.py')
        f.run(script='funC.py')


class history:

    def __init__(self,name):
        self.history=pandas.read_csv(filepath_or_buffer='../Data/'+name+'.csv')
        self.history.index=self.history['Date']
        print(f'Data from '+ str(min(self.history['Date'].values))+' to '+str(max(self.history['Date'].values)))
    
    def MA(self,day=7):
        start_date=str(datetime.datetime.strptime(min(self.history['Date'].values), "%Y-%m-%d")+ datetime.timedelta(days=day))
        end_date=max(self.history['Date'].values)
        self.history_MA=(((self.history).rolling(window=day).mean()).loc[start_date:end_date, :])

    def derivative_rate(self,order=2):
        self.derivative_rate=[]
        df=self.history.loc[:, self.history.columns != 'Date']
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
            tmpr.index=(self.history.index)[0:(n-2)]
            self.derivative_rate.append(tmpr)

    def prediction_RNN_black_box(self,data,days_ahead=4):
        X_train_list=[]
        X_train=[]
        i=0
        for history in data:
            i+=1
            n=history.history.shape[0]
            X_train_list.append(history.history.iloc[0:(n-6),])
            for derivative in history.derivative_rate:
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
        model.fit(X_train,self.history,epochs=25,batch_size=32)
            
                
            
RUI_PA=history(name='RUI.PA')
VPK_AS=history(name='VPK.AS')
BP_L=history(name="BP.L")
SHELL_AS=history(name="SHELL.AS")
TTE_PA=history(name="TTE.PA")
XOM=history(name="XOM")

RUI_PA.MA()
VPK_AS.MA()
BP_L.MA()
SHELL_AS.MA()
TTE_PA.MA()
XOM.MA()

RUI_PA.derivative_rate()
VPK_AS.derivative_rate()
BP_L.derivative_rate()
SHELL_AS.derivative_rate()
TTE_PA.derivative_rate()
XOM.derivative_rate()


#RUI_PA.prediction_NN_black_box(data=[VPK_AS,BP_L,SHELL_AS,TTE_PA,XOM])

"""
BP_L.derivative_rate()
SHELL_AS.derivative_rate()
TTE_PA.derivative_rate()
XOM.derivative_rate()"""

import pandas_datareader

start=datetime.datetime(2020,1,1)
end=datetime.datetime(2022,1,1)

data=pandas_datareader.DataReader("MRNA","yahoo",start,end)
print(data)
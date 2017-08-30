import cntk

def LoadData(fn):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("Input",45)
    dataout=cntk.io.StreamDef("Label",3)
    dataall=cntk.io.StreamDefs(labels=dataout,features=datainp)
    st=cntk.io.CTFDeserializer(n,dataall)
    rez=cntk.io.MinibatchSource(st)
    return rez

def nn():
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(100,activation=cntk.tanh)
    m=cntk.layers.Dense(3,activation=cntk.softmax)
    return m

data=LoadData("train.txt")
print(data.streams.items())
net=nn()
g=input("Нажмите любую клавишу")


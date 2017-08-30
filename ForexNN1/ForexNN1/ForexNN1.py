import cntk

def LoadData(fn):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("Input",45)
    dataout=cntk.io.StreamDef("Label",3)
    dataall=cntk.io.StreamDefs(labels=dataout,features=datainp)
    st=cntk.io.CTFDeserializer(n,dataall)
    rez=cntk.io.MinibatchSource(st)
    return rez

def nn(x):
    m=cntk.layers.Dense(100,activation=cntk.tanh)(x)
    for i in range(0,9):
        m=cntk.layers.Dense(100,activation=cntk.tanh)(m)
    m=cntk.layers.Dense(3,activation=cntk.softmax)(m)
    return m

data=LoadData("train.txt")
x=cntk.input_variable(45)
output=cntk.input_variable(3)
net=nn(x)
g=input("Нажмите любую клавишу")


import cntk

def LoadData(fn):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("Input",45)
    dataout=cntk.io.StreamDef("Label",3)
    dataall=cntk.io.StreamDefs(labels=dataout,features=datainp)
    st=cntk.io.CTFDeserializer(n,dataall)
    rez=cntk.io.MinibatchSource(st)
    return rez

data=LoadData("train.txt")

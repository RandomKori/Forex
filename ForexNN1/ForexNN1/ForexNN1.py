import os
import cntk

def LoadData(fn):
    n=os.curdir+"\\Data\\"+fn
    datainp=cntk.io.StreamDef("Input",45)
    dataout=cntk.io.StreamDef("Label",3)
    dataall=cntk.io.StreamDefs(label=dataout,futures=datainp)
    rez=cntk.io.CBFDeserializer(n,dataall)
    return rez

data=LoadData("train.txt")


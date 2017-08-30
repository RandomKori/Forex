import os
import cntk

def LoadData(fn):
    n=os.curdir+"\\Data\\"+fn
    label_dim=[]
    input_dim=[]
    data=cntk.io.StreamDef(None,(45,3),mlf=["Input","Ladel"])
    rez=cntk.io.HTKMLFDeserializer(n,data)
    return rez

data=LoadData("train.txt")

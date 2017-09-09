from __future__ import print_function
import numpy as np
import cntk
from cntk.ops.functions import load_model

def LoadData(fn,is_training):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("features",30)
    dataall=cntk.io.StreamDefs(features=datainp)
    st=cntk.io.CTFDeserializer(n,dataall)
    mbs=cntk.io.MinibatchSource(st,randomize = is_training,max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return mbs

def feval(streamf):
    z=load_model(".\\Model\\model.cmf")
    input_map={
        z.arguments[0] : streamf.streams.features,     
    }
    dat1=streamf.next_minibatch(32,input_map = input_map)
    output=z.eval(dat1)
    for i in range(len(output)):
        print(output[i])

data=LoadData("eval.txt",True)
feval(data)
g=input("Нажмите любую клавишу")
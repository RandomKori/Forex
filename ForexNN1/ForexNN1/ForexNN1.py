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
def train(net,stream):
    label=cntk.input_variable(3)
    loss=cntk.cross_entropy_with_softmax(net,label)
    label_error=cntk.classification_error(net,label)
    learning_rate=0.2
    lr_schedule=cntk.learning_rate_schedule(learning_rate,cntk.UnitType.minibatch)
    learner=cntk.sgd(net.parameters,lr_schedule)
    trainer=cntk.Trainer(net,(loss,label_error),[learner])
    return

data=LoadData("train.txt")
x=cntk.input_variable(45)
net=nn(x)
train(net,data)
g=input("Нажмите любую клавишу")


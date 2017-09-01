from __future__ import print_function
import numpy as np
import cntk

def LoadData(fn,is_training):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("features",45)
    dataout=cntk.io.StreamDef("labels",3)
    dataall=cntk.io.StreamDefs(features=datainp,labels=dataout)
    st=cntk.io.CTFDeserializer(n,dataall)
    mbs=cntk.io.MinibatchSource(st,randomize = is_training,max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return mbs

def nn(x):
    m=cntk.layers.Dense(100,activation=cntk.tanh)(x)
    for i in range(0,9):
        m=cntk.layers.Dense(100,activation=cntk.tanh)(m)
    m=cntk.layers.Dense(3,activation=cntk.softmax)(m)
    return m

def train(streamf):
    input_var = cntk.input_variable(45,np.float32, name = 'features')
    label_var=cntk.input_variable(3,np.float32, name = 'labels')
    net=nn(input_var)
    loss=cntk.cross_entropy_with_softmax(net,label_var)
    label_error=cntk.classification_error(net,label_var)
    learning_rate=0.02
    lr_schedule=cntk.learning_rate_schedule(learning_rate,cntk.UnitType.minibatch)
    learner=cntk.sgd(net.parameters,lr_schedule)
    progres=cntk.logging.ProgressPrinter(0)
    trainer=cntk.Trainer(net,(loss,label_error),[learner],progress_writers=progres)
    input_map={
        input_var : streamf.streams.features,
        label_var : streamf.streams.labels
        
    }
    minibatch_size = 50000
    num_samples_per_sweep = 1000
    for i in range(0,num_samples_per_sweep):
        dat1=streamf.next_minibatch(minibatch_size,input_map = input_map)
        trainer.train_minibatch(dat1)
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if training_loss<0.002:
            break
    return trainer

def test(streamf,trainer):
    model=trainer.model
    mb = streamf.next_minibatch(1000)
    output = model.eval(mb[streamf.streams.features])
    print(output)
    lsb=mb[streamf.streams.labels].data.asarray()
    for i in range(0,1000):
        for j in range(0,3):
            if output[i,j]>0.5:
                output[i,j]=1.0
            else:
                output[i,j]=0.0
    err=0
    for i1 in range(0,1000):
        for j1 in range(0,3):
            if lsb[i1,0,j1]!=output[i1,j1]:
                err=err+1
    print("Test error: {0:.4f}".format(err/1000*100))

data=LoadData("train.txt",True)
model1=train(data)
model1.save_checkpoint(".\\Model\\model.cmf")
data1=LoadData("test.txt",False)
test(data1,model1)
g=input("Нажмите любую клавишу")


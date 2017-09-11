from __future__ import print_function
import numpy as np
import cntk
from cntk.ops.functions import load_model

def LoadData(fn,is_training):
    n=".\\Data\\"+fn
    datainps=cntk.io.StreamDef("spread",10)
    datainph=cntk.io.StreamDef("high",10)
    datainpl=cntk.io.StreamDef("low",10)
    datainpv=cntk.io.StreamDef("volume",10)
    dataout=cntk.io.StreamDef("labels",3,is_sparse=True)
    dataall=cntk.io.StreamDefs(spread=datainps,high=datainph,low=datainpl,volume=datainpv,labels=dataout)
    st=cntk.io.CTFDeserializer(n,dataall)
    mbs=cntk.io.MinibatchSource(st,randomize = is_training,max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return mbs

def nn(s,h,l,v):
    m=cntk.layers.Stabilizer()(s)
    for i in range(0,5):
        m=cntk.layers.Recurrence(cntk.layers.LSTM(50,activation=cntk.sigmoid,init_bias=0.1,enable_self_stabilization=True))(m)
    m=cntk.splice(m,h)
    m=cntk.layers.Stabilizer()(m)
    for i in range(0,5):
        m1=cntk.layers.Recurrence(cntk.layers.LSTM(50,activation=cntk.sigmoid,init_bias=0.1,enable_self_stabilization=True))(m)
    m=cntk.splice(m,l)
    m=cntk.layers.Stabilizer()(m)
    for i in range(0,5):
        m2=cntk.layers.Recurrence(cntk.layers.LSTM(50,activation=cntk.sigmoid,init_bias=0.1,enable_self_stabilization=True))(m)
    m=cntk.splice(m,v)
    m=cntk.layers.Stabilizer()(m)
    for i in range(0,5):
        m=cntk.layers.Recurrence(cntk.layers.LSTM(50,activation=cntk.sigmoid,init_bias=0.1,enable_self_stabilization=True))(m)
    m=cntk.sequence.last(m)
    m=cntk.layers.Dense(3,activation=cntk.softmax)(m)
    return m

input_s = cntk.input_variable(10,np.float32, name = 'spread',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())
input_h = cntk.input_variable(10,np.float32, name = 'high',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())
input_l = cntk.input_variable(10,np.float32, name = 'low',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())
input_v = cntk.input_variable(10,np.float32, name = 'volume',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())
label_var=cntk.input_variable(3,np.float32, name = 'labels',is_sparse=True)


def train(streamf):
    global net
    minibatch_size =  512
    max_epochs = 2000
    epoch_size = 48985
    net=nn(input_s,input_h,input_l,input_v)
    loss = cntk.losses.cross_entropy_with_softmax(net,label_var)
    error=cntk.classification_error(net,label_var)
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule=cntk.learning_rate_schedule(lr_per_minibatch,cntk.UnitType.minibatch)
    momentum_as_time_constant = cntk.momentum_as_time_constant_schedule(700)
    learner=cntk.fsadagrad(net.parameters,lr_schedule,momentum_as_time_constant)
    progres=cntk.logging.ProgressPrinter(0)
    trainer=cntk.Trainer(net,(loss,error),[learner],progress_writers=progres)
    input_map={
        input_s : streamf.streams.spread,
        input_h : streamf.streams.high,
        input_l : streamf.streams.low,
        input_v : streamf.streams.volume,
        label_var : streamf.streams.labels
        
    }
    t = 0
    for epoch in range(max_epochs):
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end: 
            dat1=streamf.next_minibatch(minibatch_size,input_map = input_map)
            trainer.train_minibatch(dat1)
            t += dat1[label_var].num_samples
    trainer.summarize_training_progress()
    return trainer

def test(streamf):
    input_map={
        input_s : streamf.streams.spread,
        input_h : streamf.streams.high,
        input_l : streamf.streams.low,
        input_v : streamf.streams.volume,
        label_var : streamf.streams.labels   
    }
    minibatch_size =  32
    loss = cntk.losses.cross_entropy_with_softmax(net,label_var)
    progress_printer = cntk.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)
    evaluator = cntk.eval.Evaluator(loss, progress_printer)
    while True:
        dat1=streamf.next_minibatch(minibatch_size,input_map = input_map)
        if not dat1:
            break
        evaluator.test_minibatch(dat1)
    evaluator.summarize_test_progress()

def feval(streamf):
    z=load_model(".\\Model\\model.cmf")
    input_map={
        z.arguments[0] : streamf.streams.spread, 
        z.arguments[1] : streamf.streams.high,
        z.arguments[2] : streamf.streams.low,
        z.arguments[3] : streamf.streams.volume
    }
    dat1=streamf.next_minibatch(1000,input_map = input_map)
    output=z.eval(dat1)
    for i in range(len(output)):
        print(output[i])


data=LoadData("train.txt",True)
model1=train(data)
md=model1.model
md.save(".\\Model\\model.cmf")
print("========================")
data1=LoadData("test.txt",False)
test(data1)
data2=LoadData("test.txt",False)
feval(data2)
g=input("Нажмите любую клавишу")

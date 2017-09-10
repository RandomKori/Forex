from __future__ import print_function
import numpy as np
import cntk
from cntk.ops.functions import load_model

def LoadData(fn,is_training):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("features",45)
    dataout=cntk.io.StreamDef("labels",3,is_sparse=True)
    dataall=cntk.io.StreamDefs(features=datainp,labels=dataout)
    st=cntk.io.CTFDeserializer(n,dataall)
    mbs=cntk.io.MinibatchSource(st,randomize = False,max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return mbs

def nn(x):
    m=cntk.layers.Stabilizer()(x)
    for i in range(0,5):
         m=cntk.layers.Recurrence(cntk.layers.LSTM(150))(m)
    m=cntk.layers.Recurrence(cntk.layers.LSTM(3))(m)
    return m

input_var = cntk.input_variable(45,np.float32, name = 'features',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())
label_var=cntk.input_variable(3,np.float32, name = 'labels',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())


def train(streamf):
    global net
    minibatch_size =  64
    max_epochs = 3000
    epoch_size = 48985
    net=nn(input_var)
    loss = cntk.losses.cross_entropy_with_softmax(net,label_var)
    error=cntk.classification_error(net,label_var)
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule=cntk.learning_rate_schedule(lr_per_minibatch,cntk.UnitType.minibatch)
    momentum_as_time_constant = cntk.momentum_as_time_constant_schedule(700)
    learner=cntk.adam(net.parameters,lr_schedule,momentum=momentum_as_time_constant,gradient_clipping_threshold_per_sample=15,gradient_clipping_with_truncation=True)
    progres=cntk.logging.ProgressPrinter(0)
    trainer=cntk.Trainer(net,(loss,error),[learner],progress_writers=progres)
    input_map={
        input_var : streamf.streams.features,
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
        input_var : streamf.streams.features,
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
        z.arguments[0] : streamf.streams.features,     
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

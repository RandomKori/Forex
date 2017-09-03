from __future__ import print_function
import numpy as np
import cntk

def LoadData(fn,is_training):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("features",15)
    dataout=cntk.io.StreamDef("labels",1)
    dataall=cntk.io.StreamDefs(features=datainp,labels=dataout)
    st=cntk.io.CTFDeserializer(n,dataall)
    mbs=cntk.io.MinibatchSource(st,randomize = is_training,max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return mbs

def nn(x):
    with cntk.layers.default_options(initial_state = 0.1):
        m = cntk.layers.Recurrence(cntk.layers.LSTM(15))(x)
        m = cntk.sequence.last(m)
        m = cntk.layers.Dropout(0.2)(m)
        m = cntk.layers.Dense(1)(m)
    return m

def train(streamf):
    input_var = cntk.input_variable(15,np.float32, name = 'features',dynamic_axes=cntk.axis.Axis.default_input_variable_dynamic_axes())
    label_var=cntk.input_variable(1,np.float32, name = 'labels')
    net=nn(input_var)
    loss = cntk.squared_error(net,label_var)
    error=cntk.squared_error(net,label_var)
    learning_rate=0.002
    lr_schedule=cntk.learning_rate_schedule(learning_rate,cntk.UnitType.minibatch)
    momentum_time_constant = cntk.momentum_as_time_constant_schedule(5000 / -np.math.log(0.9))
    learner=cntk.fsadagrad(net.parameters,lr=lr_schedule,momentum = momentum_time_constant,unit_gain = True)
    progres=cntk.logging.ProgressPrinter(0)
    trainer=cntk.Trainer(net,(loss,error),[learner],progress_writers=progres)
    input_map={
        input_var : streamf.streams.features,
        label_var : streamf.streams.labels
        
    }
    minibatch_size = 5000
    num_samples_per_sweep = 5000
    for i in range(0,num_samples_per_sweep):
        dat1=streamf.next_minibatch(minibatch_size,input_map = input_map)
        trainer.train_minibatch(dat1)
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
    return trainer

def test(streamf,trainer):
    model=trainer.model
    mb = streamf.next_minibatch(1000)
    output = model.eval(mb[streamf.streams.features])
    lsb=mb[streamf.streams.labels].data.asarray()
    for i in range(0,1000):
        print("[ {0} ]".format(output[i,0]))

data=LoadData("train.txt",True)
model1=train(data)
md=model1.model
md.save(".\\Model\\model.cmf")
data1=LoadData("test.txt",False)
test(data1,model1)
g=input("Нажмите любую клавишу")


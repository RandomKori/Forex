from __future__ import print_function
import numpy as np
import cntk

def LoadData(fn,is_training):
    n=".\\Data\\"+fn
    datainp=cntk.io.StreamDef("features",45)
    dataout=cntk.io.StreamDef("labels",3)
    dataall=cntk.io.StreamDefs(features=datainp,labels=dataout)
    st=cntk.io.CTFDeserializer(n,dataall)
    rez=cntk.io.MinibatchSource(st,randomize = is_training,max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return rez

def nn(x):
    m=cntk.layers.Dense(100,activation=cntk.tanh)(x)
    for i in range(0,9):
        m=cntk.layers.Dense(100,activation=cntk.tanh)(m)
    m=cntk.layers.Dense(3,activation=cntk.softmax)(m)
    return m

def train(streamf):
    input_var = cntk.input_variable(45)
    label_var=cntk.input_variable(3)
    net=nn(input_var)
    loss=cntk.cross_entropy_with_softmax(net,label_var)
    label_error=cntk.classification_error(net,label_var)
    learning_rate=0.2
    lr_schedule=cntk.learning_rate_schedule(learning_rate,cntk.UnitType.minibatch)
    learner=cntk.sgd(net.parameters,lr_schedule)
    trainer=cntk.Trainer(net,(loss,label_error),learner)
    input_map={
        input_var : streamf.streams.features,
        label_var : streamf.streams.labels
    }
    training_progress_output_freq = 500
    minibatch_size = 1
    num_samples_per_sweep = 10000
    for i in range(0,num_samples_per_sweep):
        dat1=streamf.next_minibatch(minibatch_size,input_map = input_map)
        trainer.train_minibatch(dat1)
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(i, training_loss, eval_error*100))
        if training_loss<0.002:
            break
    return trainer

data=LoadData("train.txt",True)
model1=train(data)
model1.save_checkpoint(".\\Model\\model.crnf")
g=input("Нажмите любую клавишу")


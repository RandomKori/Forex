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
    trainer=cntk.Trainer(net,(loss,label_error),[learner])
    input_map={
        input_var : streamf.streams.features,
        label_var : streamf.streams.labels
        
    }
    minibatch_size = 5000
    num_samples_per_sweep = 1000
    for i in range(0,num_samples_per_sweep):
        dat1=streamf.next_minibatch(minibatch_size,input_map = input_map)
        trainer.train_minibatch(dat1)
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(i, training_loss, eval_error*100))
        if training_loss<0.002:
            break
    return trainer

def test(streamf,trainer):
    test_input_var = cntk.input_variable(45,np.float32, name = 'features')
    test_label_var=cntk.input_variable(3,np.float32, name = 'labels')
    test_input_map={
        test_input_var : streamf.streams.features,
        test_label_var : streamf.streams.labels
    }
    test_minibatch_size=1000
    test_result = 0.0
    for i in range(10):
        dat1=streamf.next_minibatch(test_minibatch_size,input_map = test_input_map)
        eval_error=trainer.test_minibatch(dat1)
        test_result = test_result + eval_error
    print("Average test error: {0:.2f}%".format(eval_error*100)/10)
    return

data=LoadData("train.txt",True)
model1=train(data)
model1.save_checkpoint(".\\Model\\model.cmf")
data1=LoadData("test.txt",False)
test(data1,model1)
g=input("Нажмите любую клавишу")


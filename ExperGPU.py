#INIT COLAB
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

bigpicklepath=r'C:\Users\sidch\Desktop\bigpickle'
testpicklepath=r'C:\Users\sidch\Desktop\testpickle'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Embedding, SimpleRNN, GRU
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import pickle
import copy
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# fix random seed for reproducibility
np.random.seed(21)
from time import process_time 


try:
    shuffle_arr
except NameError:
    try:
        list_of_data_arrays
    except NameError:
        with open(bigpicklepath, "rb") as fp:   # Unpickling
            list_of_data_arrays=pickle.load(fp)    

    shuffle=list_of_data_arrays
    random.shuffle(shuffle)
    del(list_of_data_arrays)

    shuffle_arr=np.array(shuffle)[0:,0]
    shuffle_labels=np.array(shuffle)[0:,1]
    shuffle_obj_ids=np.array(shuffle)[0:,2]
    del(shuffle)

try:
    list_of_data_arrays2
except NameError:
    with open(testpicklepath, "rb") as fp:   # Unpickling
        list_of_data_arrays2=pickle.load(fp)    

shuffle2=list_of_data_arrays2
random.shuffle(shuffle2)
del(list_of_data_arrays2)

shuffle_arr2=np.array(shuffle2)[0:,0]
shuffle_labels2=np.array(shuffle2)[0:,1]
shuffle_obj_ids2=np.array(shuffle2)[0:,2]
del(shuffle2)

# Define inputs and targets
max_length=max(map(len,shuffle_arr))
timesteps=max_length
data_dim = shuffle_arr[0].shape[1]
num_classes = 15


inputs = sequence.pad_sequences(shuffle_arr, maxlen=max_length,padding='post')
targets = to_categorical(shuffle_labels, num_classes=15)
test_inputs = sequence.pad_sequences(shuffle_arr2, maxlen=max_length,padding='post')
test_targets = to_categorical(shuffle_labels2, num_classes=15)
inputs = inputs.astype(np.float32)
test_inputs = test_inputs.astype(np.float32)

tot_samples=len(targets)

#Some constants
num_epochs=20
loss='categorical_crossentropy'
optimizer="adam"
activation="relu"

#Conv1D feature list
mask_status_list=[True,False]
conv1d_status_list=[True,False]
conv1d_num_layer_list=[1,2,3]
conv1d_num_filter_list=[16, 32, 64, 128, 256]
conv1d_kernel_size_list=[2,3,5]

#RNN feature list
rnn_num_layer_list=[1,2,3]
rnn_type_list=["SimpleRNN", "GRU", "LSTM"]
rnn_bidir_list=[True,False]
rnn_num_unit_list=[32, 64, 128, 256]

#Dense feature list
dense_num_layer_list=[0,1,2,3]
dense_num_unit_list=[32, 64, 128, 256]

#General features
batch_size_list=[32, 64, 128,256,512]
dropout_rate_list=[0.05, 0.1, 0.2, 0.5]

def build_model():
    t1=process_time()
    mask_status=random.choice(mask_status_list)
    conv1d_status=random.choice(conv1d_status_list)
    conv1d_num_layer=random.choice(conv1d_num_layer_list)
    conv1d_num_filter=random.choice(conv1d_num_filter_list)
    conv1d_kernel_size=random.choice(conv1d_kernel_size_list)
    rnn_num_layer=random.choice(rnn_num_layer_list)
    rnn_type=random.choice(rnn_type_list)
    rnn_bidir=random.choice(rnn_bidir_list)
    rnn_num_unit=random.choice(rnn_num_unit_list)
    dense_num_layer=random.choice(dense_num_layer_list)
    dense_num_unit=random.choice(dense_num_unit_list)
    batch_size=random.choice(batch_size_list)
    dropout_rate=random.choice(dropout_rate_list)
    name=f"mask_status = {mask_status}, conv1d_status = {conv1d_status}, conv1d_num_layer = {conv1d_num_layer}, conv1d_num_filter = {conv1d_num_filter}, conv1d_kernel_size = {conv1d_kernel_size}, rnn_num_layer = {rnn_num_layer}, rnn_type = {rnn_type}, rnn_bidir = {rnn_bidir}, rnn_num_unit = {rnn_num_unit}, dense_num_layer = {dense_num_layer}, dense_num_unit = {dense_num_unit}, batch_size = {batch_size}, dropout_rate = {dropout_rate}"
    print("************")
    print(name)
    print("************")
    done_features.append(name)
    clear_session()
    model=Sequential()
    #Masking Layer
    if mask_status:
        model.add(Masking(mask_value = 0,input_shape=(timesteps, data_dim)))
    #Conv Layer
    if conv1d_status:
        if conv1d_num_layer == 1:
            if mask_status:
                model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation))
            else:
                model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation,input_shape=(timesteps, data_dim)))
        elif conv1d_num_layer == 2:
            if mask_status:
                model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation))
            else:
                model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation,input_shape=(timesteps, data_dim)))
            model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation))
        elif conv1d_num_layer == 3:
            if mask_status:
                model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation))
            else:
                model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation,input_shape=(timesteps, data_dim)))
            model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation))
            model.add(Conv1D(filters=conv1d_num_filter, kernel_size=conv1d_kernel_size,padding='same', activation=activation))
        model.add(Dropout(dropout_rate))
    #RNN Layer
    if not rnn_bidir:
        #Non Bidirectional RNNS:
        if rnn_type=="LSTM":
            if rnn_num_layer==1:
                if not (mask_status or conv1d_status):
                    model.add(LSTM(rnn_num_unit,batch_input_shape=(tot_samples, timesteps, data_dim)))
                else:
                    model.add(LSTM(rnn_num_unit))

            if rnn_num_layer==2:
                if not (mask_status or conv1d_status):
                    model.add(LSTM(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim)))
                else:
                    model.add(LSTM(rnn_num_unit,return_sequences=True))
                model.add(LSTM(rnn_num_unit))

            if rnn_num_layer==3:
                if not (mask_status or conv1d_status):
                    model.add(LSTM(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim)))
                else:
                    model.add(LSTM(rnn_num_unit,return_sequences=True))
                model.add(LSTM(rnn_num_unit,return_sequences=True))
                model.add(LSTM(rnn_num_unit))

        elif rnn_type=="SimpleRNN":
            if rnn_num_layer==1:
                if not (mask_status or conv1d_status):
                    model.add(SimpleRNN(rnn_num_unit,input_shape=(timesteps, data_dim)))
                else:
                    model.add(SimpleRNN(rnn_num_unit))

            if rnn_num_layer==2:
                if not (mask_status or conv1d_status):
                    model.add(SimpleRNN(rnn_num_unit,return_sequences=True,input_shape=(timesteps, data_dim)))
                else:
                    model.add(SimpleRNN(rnn_num_unit,return_sequences=True))
                model.add(SimpleRNN(rnn_num_unit))

            if rnn_num_layer==3:
                if not (mask_status or conv1d_status):
                    model.add(SimpleRNN(rnn_num_unit,return_sequences=True,input_shape=(timesteps, data_dim)))
                else:
                    model.add(SimpleRNN(rnn_num_unit,return_sequences=True))
                model.add(SimpleRNN(rnn_num_unit,return_sequences=True))
                model.add(SimpleRNN(rnn_num_unit))

        elif rnn_type=="GRU":
            if rnn_num_layer==1:
                if not (mask_status or conv1d_status):
                    model.add(GRU(rnn_num_unit,batch_input_shape=(tot_samples, timesteps, data_dim)))
                else:
                    model.add(GRU(rnn_num_unit))

            if rnn_num_layer==2:
                if not (mask_status or conv1d_status):
                    model.add(GRU(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim)))
                else:
                    model.add(GRU(rnn_num_unit,return_sequences=True))
                model.add(GRU(rnn_num_unit))

            if rnn_num_layer==3:
                if not (mask_status or conv1d_status):
                    model.add(GRU(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim)))
                else:
                    model.add(GRU(rnn_num_unit,return_sequences=True))
                model.add(GRU(rnn_num_unit,return_sequences=True))
                model.add(GRU(rnn_num_unit))

    else:   
        #Bidirectional RNNs
        if rnn_type=="LSTM":
            if rnn_num_layer==1:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(LSTM(rnn_num_unit,batch_input_shape=(tot_samples, timesteps, data_dim))))
                else:
                    model.add(Bidirectional(LSTM(rnn_num_unit)))

            if rnn_num_layer==2:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(LSTM(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim))))
                else:
                    model.add(Bidirectional(LSTM(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(LSTM(rnn_num_unit)))

            if rnn_num_layer==3:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(LSTM(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim))))
                else:
                    model.add(Bidirectional(LSTM(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(LSTM(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(LSTM(rnn_num_unit)))

        elif rnn_type=="SimpleRNN":
            if rnn_num_layer==1:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(SimpleRNN(rnn_num_unit,input_shape=(timesteps, data_dim))))
                else:
                    model.add(Bidirectional(SimpleRNN(rnn_num_unit)))

            if rnn_num_layer==2:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(SimpleRNN(rnn_num_unit,return_sequences=True,input_shape=(timesteps, data_dim))))
                else:
                    model.add(Bidirectional(SimpleRNN(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(SimpleRNN(rnn_num_unit)))

            if rnn_num_layer==3:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(SimpleRNN(rnn_num_unit,return_sequences=True,input_shape=(timesteps, data_dim))))
                else:
                    model.add(Bidirectional(SimpleRNN(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(SimpleRNN(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(SimpleRNN(rnn_num_unit)))

        elif rnn_type=="GRU":
            if rnn_num_layer==1:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(GRU(rnn_num_unit,batch_input_shape=(tot_samples, timesteps, data_dim))))
                else:
                    model.add(Bidirectional(GRU(rnn_num_unit)))

            if rnn_num_layer==2:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(GRU(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim))))
                else:
                    model.add(Bidirectional(GRU(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(GRU(rnn_num_unit)))

            if rnn_num_layer==3:
                if not (mask_status or conv1d_status):
                    model.add(Bidirectional(GRU(rnn_num_unit,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim))))
                else:
                    model.add(Bidirectional(GRU(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(GRU(rnn_num_unit,return_sequences=True)))
                model.add(Bidirectional(GRU(rnn_num_unit)))

    model.add(Dropout(dropout_rate))
    #Dense Layer
    if dense_num_layer == 1:
        model.add(Dense(dense_num_unit, activation=activation))
    elif dense_num_layer == 2:
        model.add(Dense(dense_num_unit, activation=activation))
        model.add(Dense(dense_num_unit, activation=activation))
    elif dense_num_layer == 3:
        model.add(Dense(dense_num_unit, activation=activation))
        model.add(Dense(dense_num_unit, activation=activation))
        model.add(Dense(dense_num_unit, activation=activation))
    if dense_num_layer != 0:
        model.add(Dropout(dropout_rate))
        
    #Final layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    history=model.fit(inputs, targets,batch_size=batch_size, epochs=num_epochs,shuffle=True, verbose=1)
    tr_loss=history.history["loss"][-1]
    tr_acc=history.history["accuracy"][-1]
    evalhis = model.evaluate(test_inputs,test_targets, verbose=True)
    eval_loss=evalhis[0]
    eval_acc=evalhis[1]
    res=f"Training Loss = {tr_loss}, Training Accuracy = {tr_acc}, Evaluation Loss = {eval_loss}, Evaluation Accuracy = {eval_acc}"
    result_per_model.append(res)
    t2=process_time()
    print(f"Took {t2 - t1} s to run this model.")

with open(r"C:\Users\sidch\Desktop\done_features_pickle", "rb") as fp:   # Unpickling
    done_features=pickle.load(fp)    
with open(r"C:\Users\sidch\Desktop\result_per_model_pickle", "rb") as fp:   # Unpickling
    result_per_model=pickle.load(fp) 
    
assert len(done_features)==len(result_per_model)
print(f"Found {len(done_features)} previous experiments. Cool.")
oglen=len(done_features)


def save_pickles():
    assert len(done_features)==len(result_per_model) and len(done_features)>=oglen
    with open(r"C:\Users\sidch\Desktop\done_features_pickle", "wb") as fp:   #Pickling
        pickle.dump(done_features, fp)
    with open(r"C:\Users\sidch\Desktop\result_per_model_pickle", "wb") as fp:   #Pickling
        pickle.dump(result_per_model, fp)
        
errorlog=[]

for i in range(2):
    print(f"Iteration {i+1}")
    try:
        build_model()
        print(result_per_model[-1])
        if len(done_features)==len(result_per_model):
            save_pickles()
    except Exception as e:
        print(e)
        errorlog.append(e)
        if len(done_features)-len(result_per_model)==1:
            del done_features[-1]
            save_pickles()
            continue
        assert len(done_features)==len(result_per_model)
        continue

print(str(errorlog))
print("Done")
#for %i IN (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50) DO python ExperGPU.py
# os.system('cmd /k "psshutdown -d -t 0"') 
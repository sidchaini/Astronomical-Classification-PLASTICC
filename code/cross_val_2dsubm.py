from scsskutils import *
import os

loc_train_2dpickle=os.path.join(pickle_location,"training_set_2d_pickle")
loc_train_3dpickle=os.path.join(pickle_location,"training_set_3d_pickle")
loc_train_labels=os.path.join(pickle_location,"training_set_label_pickle")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Embedding, SimpleRNN, GRU,SpatialDropout1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Flatten, Masking, GaussianNoise, concatenate, Embedding, TimeDistributed
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import kerastuner as kt
from kerastuner import HyperModel
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.compat.v1 import InteractiveSession
# config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# fix random seed for reproducibility
np.random.seed(21)
from time import process_time 
# from livelossplot.tf_keras import PlotLossesCallback
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [PlotLossesCallback(), checkpoint]
es = EarlyStopping(monitor='val_loss',mode="min",verbose=1,patience=100)
callbacks_list = [checkpoint,es]


with open(loc_train_2dpickle, "rb") as fp:   # Unpickling
    train_arr_2d=pickle.load(fp)

with open(loc_train_labels, "rb") as fp:   # Unpickling
    train_labels=pickle.load(fp)

num_classes = 15
inputs2d = np.array(train_arr_2d)
del(train_arr_2d)
targets = to_categorical(train_labels, num_classes=15)
del(train_labels)
inputs2d = inputs2d.astype(np.float32)
targets = targets.astype(np.int32)

# timesteps,data_dim = inputs3d.shape[1:3] 
dim_2d_feats=inputs2d.shape[1]
tot_samples = len(inputs2d)

def plot_model_change(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def get_model():
    clear_session()
    inputs_2d = Input(shape=dim_2d_feats,)
    mod = Dense(512,activation="tanh")(inputs_2d)
    mod = Dropout(0.1)(mod)
    mod = BatchNormalization()(mod)

    mod = Dense(128, activation="tanh")(mod)
    mod = Dropout(0.1)(mod)
    mod = BatchNormalization()(mod)

    mod = Dense(128, activation="tanh")(mod)
    mod = Dropout(0.1)(mod)
    mod = BatchNormalization()(mod)

    mod = Dense(64, activation="tanh")(mod)
    mod = Dropout(0.1)(mod)
    mod = BatchNormalization()(mod)


    output_layer = Dense(num_classes, activation="softmax") (mod)
    model = Model(inputs=inputs_2d, outputs=output_layer, name="SCSSK_dense_only")
    model.compile(loss="categorical_crossentropy",optimizer=Adam(1e-2),metrics=['accuracy'])
    return model

kf = KFold(n_splits=5,shuffle=True,random_state=44)

X = inputs2d
y = targets
i=0
for train_index, val_index in kf.split(X):
    i+=1
    print("****************************")
    print(f"Fold {i}: Tot Train:", len(train_index), "Tot Val:", len(val_index))
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    checkpoint = ModelCheckpoint(f"best_model_fold{i}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss',mode="min",verbose=1,patience=20)
    callbacks_list = [checkpoint,es]
    model = get_model()
    history = model.fit(X_train,y_train,batch_size=1000,verbose=True,epochs=600,validation_data=(X_val,y_val),callbacks=callbacks_list)
    plot_model_change(history)
    model = load_model(f"best_model_fold{i}.h5")
    print("Loss, Accuracy")
    result=model.evaluate(X_val, y_val,verbose=True,batch_size=500)
    print(result)
    print("****************************")
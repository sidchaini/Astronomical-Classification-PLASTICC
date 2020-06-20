'''
Note: We performed quite a lot of random searches using Keras-Tuner, and have not included all of them.
This py file is the one which generated the best result for 2dsubm models.
'''
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
from tensorflow.keras.callbacks import ModelCheckpoint
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
callbacks_list = [checkpoint]

with open(loc_train_3dpickle, "rb") as fp:   # Unpickling
    train_arr_3d=pickle.load(fp)

# with open(loc_train_2dpickle, "rb") as fp:   # Unpickling
#     train_arr_2d=pickle.load(fp)

with open(loc_train_labels, "rb") as fp:   # Unpickling
    train_labels=pickle.load(fp)

max_length=max(map(len,train_arr_3d))
num_classes = 15

inputs3d = sequence.pad_sequences(train_arr_3d, maxlen=max_length,padding='post',dtype='float32')
del(train_arr_3d)
targets = to_categorical(train_labels, num_classes=15)
del(train_labels)
inputs3d = inputs3d.astype(np.float32)
targets = targets.astype(np.int32)

timesteps,data_dim = inputs3d.shape[1:3]
tot_samples = len(inputs3d)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        yc = tf.clip_by_value(y_pred,1e-15,1-1e-15)
        pt_1 = tf.where(tf.equal(y_true, 1), yc, tf.ones_like(yc))
        pt_0 = tf.where(tf.equal(y_true, 0), yc, tf.zeros_like(yc))
        return (-tf.keras.backend.sum(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1))-tf.keras.backend.sum((1-alpha) * tf.keras.backend.pow( pt_0, gamma) * tf.keras.backend.log(1. - pt_0)))
    return focal_loss_fixed

class MyHyperModel(kt.HyperModel):
   
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def build(self, hp):
        #GRUs
        inputs_gru = Input(shape=(timesteps,data_dim))
        noise_layer = GaussianNoise(hp.Choice('gaussian_noise', [0.0,0.25,0.5,0.75,1.0]))(inputs_gru)
        start_gru_units = hp.Choice('units_first_layer_gru', [64, 128, 256])        
        gru = Bidirectional(GRU(start_gru_units,return_sequences=True,batch_input_shape=(tot_samples, timesteps, data_dim))) (noise_layer)
        gru_spat_drop = 0.1
#         gru_spat_drop = hp.Choice('gru_spat_dropout', [0.0, 0.01, 0.1, 0.25,0.5])
        old_gru_units  = start_gru_units
        
        for i in range(3):
            spatialdrop = SpatialDropout1D(gru_spat_drop)(gru)
            cur_gru_units = hp.Choice('units_gru_deep'+str(i),[max(32,old_gru_units//2),old_gru_units])
            gru = Bidirectional(GRU(cur_gru_units,return_sequences=True)) (spatialdrop)
            old_gru_units = cur_gru_units

        #Pool into dense
        global_maxpool = GlobalMaxPooling1D() (gru)
        dense_act = "tanh"
        dense_drop = hp.Choice('dense_dropout', [0.0, 0.1])
        start_dense_units = hp.Choice('units_first_dense', [256,512,1024])
        dense1 = Dense(start_dense_units,activation=dense_act)(global_maxpool)

        drop = Dropout(dense_drop)(dense1)
        old_dense_units = start_dense_units
        
        for i in range(4):
            cur_dense_units = hp.Choice('units_dense_deep'+str(i),[max(32,old_dense_units//2),old_dense_units])
            dense = Dense(cur_dense_units, activation=dense_act)(drop)
            drop = Dropout(dense_drop)(dense)
            old_dense_units = cur_dense_units
        
        output_layer = Dense(num_classes, activation="softmax") (drop)
        model = Model(inputs=inputs_gru, outputs=output_layer, name="SCSSK_gru_only")
        loss_choice = hp.Choice('loss_function', ["focal","categorical_crossentropy"])
        if loss_choice == "focal":
            model.compile(loss=focal_loss(),
                          optimizer="adam",
                          metrics=['accuracy'])
        elif loss_choice == "categorical_crossentropy":
            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=['accuracy'])
        return model

hypermodel = MyHyperModel(num_classes=15)

tuner = kt.RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=2,
    directory='SCSSK',
    project_name='Hyperparameter tuning')

tuner.search(inputs3d, targets,batch_size=1000, epochs=50,shuffle=True, verbose=0, validation_split = 0.05)

models=tuner.get_best_models(num_models=20)
for i,model in enumerate(models):
    model.save(f"best_3dsubm{i+1}.h5")

tuner.results_summary()


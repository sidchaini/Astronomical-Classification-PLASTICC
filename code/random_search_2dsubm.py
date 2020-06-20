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

dim_2d_feats=inputs2d.shape[1]
tot_samples = len(inputs2d)


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
        dense_drop = hp.Choice('dense_dropout', [0.01, 0.1])
        start_dense_units = hp.Choice('units_first_dense', [256, 512, 1024])
        inputs_2d = Input(shape=dim_2d_feats,)
        mod = Dense(start_dense_units,activation="tanh")(inputs_2d)
        mod = Dropout(dense_drop)(mod)
        mod = BatchNormalization()(mod)
        old_dense_units = start_dense_units
        for i in range(hp.Int('num_dense_layers', 2, 5)):
            cur_dense_units = hp.Choice('units_dense_deep'+str(i),[max(32,old_dense_units//2),old_dense_units])
            mod = Dense(cur_dense_units, activation="tanh")(mod)
            mod = Dropout(dense_drop)(mod)
            mod = BatchNormalization()(mod)
            old_dense_units = cur_dense_units
        output_layer = Dense(num_classes, activation="softmax") (mod)
        model = Model(inputs=inputs_2d, outputs=output_layer, name="SCSSK_dense_only")
        lrn_rate = 1e-2
        loss_choice = hp.Choice('loss_function', ["focal","categorical_crossentropy"])        
        
        if loss_choice == "focal":
            model.compile(loss=focal_loss(),
                          optimizer=Adam(lrn_rate),
                          metrics=['accuracy'])
        else:
            model.compile(loss="categorical_crossentropy",
                          optimizer=Adam(lrn_rate),
                          metrics=['accuracy'])
        
        return model

hypermodel = MyHyperModel(num_classes=15)

tuner = kt.RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=3,
    directory='SCSSK',
    project_name='Dense Tuner')

tuner.search(inputs2d, targets,batch_size=1000, epochs=500,shuffle=True, verbose=0, validation_split = 0.05)

models=tuner.get_best_models(num_models=20)
for i,model in enumerate(models):
    model.save(f"best_2dsubm{i+1}.h5")

tuner.results_summary()
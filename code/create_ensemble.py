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
from sklearn import metrics
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
loc_val_2dpickle="../input/endgamepickles3/val_set_2d_pickle"
loc_val_3dpickle="../input/endgamepickles3/val_set_3d_pickle"
loc_val_labels="../input/endgamepickles3/val_set_label_pickle"


gamma=2
alpha=.25
def focal_loss_fixed(y_true, y_pred):
    yc = tf.clip_by_value(y_pred,1e-15,1-1e-15)
    pt_1 = tf.where(tf.equal(y_true, 1), yc, tf.ones_like(yc))
    pt_0 = tf.where(tf.equal(y_true, 0), yc, tf.zeros_like(yc))
    return (-tf.keras.backend.sum(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1))-tf.keras.backend.sum((1-alpha) * tf.keras.backend.pow( pt_0, gamma) * tf.keras.backend.log(1. - pt_0)))

model_dense1 = load_model("best_2dsubm_1.h5")
model_dense2 = load_model("best_2dsubm_2.h5")
model_gru1 = load_model("best_3dsubm_1.h5",custom_objects={'focal_loss_fixed': focal_loss_fixed})
model_gru2 = load_model("best_3dsubm_2.h5")


def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	output = Dense(15, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# define ensemble model
members = [model_dense1,model_dense2,model_gru1,model_gru2]            #CHANGE
stacked_model = define_stacked_model(members)



with open(loc_val_3dpickle, "rb") as fp:   # Unpickling
    val_arr_3d=pickle.load(fp)

with open(loc_val_2dpickle, "rb") as fp:   # Unpickling
    val_arr_2d=pickle.load(fp)

with open(loc_val_labels, "rb") as fp:   # Unpickling
    val_labels=pickle.load(fp)

max_length = 162
num_classes = 15


val_inputs3d = sequence.pad_sequences(val_arr_3d, maxlen=max_length,padding='post',dtype='float32')
del(val_arr_3d)
val_inputs2d = np.array(val_arr_2d)
del(val_arr_2d)
val_targets = to_categorical(val_labels, num_classes=15)
del(val_labels)
val_inputs3d = val_inputs3d.astype(np.float32)
val_inputs2d = val_inputs2d.astype(np.float32)
val_targets = val_targets.astype(np.int32)

timesteps,data_dim = (162, 13)
dim_2d_feats=156
tot_samples = 7848

history = stacked_model.fit([val_inputs2d,val_inputs2d,val_inputs3d,val_inputs3d], val_targets, epochs=50, verbose=1)                  #CHANGE

stacked_model.save("fourstacked_model.h5")

def plot_model_change(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

plot_model_change(history)

preds = stacked_model.predict([test_inputs2d,test_inputs2d,test_inputs3d,test_inputs3d],verbose=True)                 #CHANGE
y_preds = preds.argmax(axis=1)
y_preds_2 = preds
y_true = test_targets.argmax(axis=1)

print(metrics.accuracy_score(y_true, y_preds))

def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

import pandas as pd
print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true, y_preds_2))

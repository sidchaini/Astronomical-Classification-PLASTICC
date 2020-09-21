import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Embedding, SimpleRNN, GRU
from tensorflow.keras.layers import GlobalMaxPooling1D, Flatten, Masking, GaussianNoise, concatenate
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from sklearn.model_selection import KFold
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import accuracy_score

from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# fix random seed for reproducibility
np.random.seed(21)
from time import process_time 

from scsskutils import *

print("Loading Model...")
model = load_model("fourstacked_model.h5")
print("Loaded Model Successfully.")

max_length = 162
num_classes = 15
timesteps,data_dim = (162, 13)
dim_2d_feats=156
# tot_samples = 7848

for i in range(1,12):
    print(f"************ test_batch_{i} ************")
    print("Loading all necessary Pickles...")
    loc_test_3dpickle=os.path.join(pickle_location,f"test_set_batch{i}_3d_pickle")
    loc_test_2dpickle=os.path.join(pickle_location,f"test_set_batch{i}_2d_pickle")
    loc_test_labels = os.path.join(pickle_location,f"test_set_batch{i}_label_pickle")
    if (os.path.isfile(f"preds{i}.csv")
       ):
        boolchoice=query_yes_no(f"preds{i}.csv found! Do you want to predict again?")
        if boolchoice==False:
            continue
    with open(loc_test_3dpickle, "rb") as fp:   # Unpickling
        test_arr_3d=pickle.load(fp)

    with open(loc_test_2dpickle, "rb") as fp:   # Unpickling
        test_arr_2d=pickle.load(fp)

    with open(loc_test_labels, "rb") as fp:   # Unpickling
        test_labels=pickle.load(fp)
    print("Loaded Pickles Successfully.")

    test_inputs3d = sequence.pad_sequences(test_arr_3d, maxlen=max_length,padding='post',dtype='float32')
    del(test_arr_3d)
    test_inputs2d = np.array(test_arr_2d)
    del(test_arr_2d)
    test_targets = to_categorical(test_labels, num_classes=15)
    del(test_labels)
    test_inputs3d = test_inputs3d.astype(np.float32)
    test_inputs2d = test_inputs2d.astype(np.float32)
    test_targets = test_targets.astype(np.int32)

    print("Predicting Results...")
    # model.evaluate([test_inputs2d,test_inputs2d,test_inputs3d,test_inputs3d],test_targets,verbose=True,batch_size=128)
    predictions = model.predict([test_inputs2d,test_inputs2d,test_inputs3d,test_inputs3d],verbose=True, batch_size=128)
    print("Prediction complete.")

    y_preds = predictions.argmax(axis=1)
    y_true = test_targets.argmax(axis=1)
    acc = accuracy_score(y_true, y_preds)
    print('Accuracy: {1:.4f}'.format(acc))
    del(y_preds)
    del(y_true)


    df = pd.DataFrame(data=predictions,columns=['class_6', 'class_15', 'class_16', 'class_42', 'class_52',
           'class_53', 'class_62', 'class_64', 'class_65', 'class_67', 'class_88',
           'class_90', 'class_92', 'class_95', 'class_99'])
    del(predictions)
    print(f"\tSaving Result...")
    df.to_csv("preds{i}.csv", index=False)
    print("\tSaved Successfully.")

print("Program completed.")
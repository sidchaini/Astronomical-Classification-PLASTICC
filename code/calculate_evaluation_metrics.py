import numpy as np
import pandas as pd
from scsskutils import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


df=pd.read_csv("final_submission.csv")

y_preds = df.drop("object_id",1).to_numpy().argmax(axis=1)
y_preds_2 = df.drop("object_id",1).to_numpy()
del(df)
df2=pd.read_csv(os.path.join(data_location,"unblinded_test_set_metadata.csv")

target_dict ={
    6: 0,
    15: 1,
    16: 2,
    42: 3,
    52: 4,
    53: 5,
    62: 6,
    64: 7,
    65: 8,
    67: 9,
    88: 10,
    90: 11,
    92: 12,
    95: 13,
    99: 14,
    991: 14,
    992: 14,
    993: 14,
    994: 14
}
df2["target_class"]=df2.loc[:,["target"]].replace(target_dict)

y_true=df2["target_class"].values
del(df2)

acc = metrics.accuracy_score(y_true, y_preds)
precision = metrics.precision_score(y_true, y_preds,average="macro")
recall = metrics.recall_score(y_true, y_preds,average="macro")
roc=multiclass_roc_auc_score(y_true, y_preds)
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC Auc Score: {roc}")

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

mywlogloss = multi_weighted_logloss(y_true, y_preds_2)
print(f"Multi Weighted Log Loss: {mywlogloss}")

import matplotlib.pyplot as plt

plt.hist(y_true, bins = 15, label = "True")
plt.hist(y_preds, bins = 15, label = "Preds")
plt.legend()
plt.show()

import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from numpy import copy

my_dict={
     0:6,
     1:15,
    2:16,
    3:42,
    4:52,
    5:53,
    6:62,
    7:64,
    8:65,
    9:67,
    10:88,
    11:90,
    12:92,
    13:95,
    14:99,
    14:99,
    14:99,
    14:99,
    14:99
}
y_preds_s = copy(y_preds)
for k, v in my_dict.items(): y_preds_s[y_preds==k] = v

y_true_s = copy(y_true)
for k, v in my_dict.items(): y_true_s[y_true==k] = v

plot = skplt.metrics.plot_confusion_matrix(y_true_s, y_preds_s,figsize=(12,12),normalize=True)
fig = plot.get_figure()
fig.savefig("confusion_matrix.png",dpi=300)

plot = skplt.metrics.plot_confusion_matrix(y_true_s, y_preds_s,figsize=(12,12),normalize=False)
fig = plot.get_figure()
fig.savefig("confusion_matrix_alt.png",dpi=300)

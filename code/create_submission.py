frames=[]
from scsskutils import *

import os
import numpy as np
import pandas as pd

for i in range(1,12):
    df=pd.read_csv(fr"preds{i}.csv")
    frames.append(df)

df = pd.concat(frames)
del frames
print("Pleasure ensure that the PLAsTiCC sample_submission.csv is located in the directory {os.getcwd()}.")
sampsubloc="sample_submission.csv"
df = df.reset_index(drop=True)
df2=pd.read_csv(sampsubloc)
objs = df2.object_id
del(df2)
df.insert(0, 'object_id', objs)
df.to_csv("final_submission.csv",index=False)
print("File is now ready to be submitted!")
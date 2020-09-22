from scsskutils import *

import pandas as pd
import numpy as np
from time import process_time 
import pickle
import random
from pathlib import Path
import os
import sys
from tqdm import tqdm
# from tqdm.notebook import tqdm
from sklearn import preprocessing
import gc
import requests
import gzip



def csv_to_pickle(mainfilename,metadatafilename):
    mainfilename=mainfilename.split(".csv")[0]
    metadatafilename=metadatafilename.split(".csv")[0]
    if (os.path.isfile(os.path.join(pickle_location,f"{mainfilename}_3d_pickle"))
        or os.path.isfile(os.path.join(pickle_location,f"{mainfilename}_2d_pickle"))
        or os.path.isfile(os.path.join(pickle_location,f"{mainfilename}_label_pickle"))
       ):
        boolchoice=query_yes_no(f"{mainfilename} pickles found! Do you want to rebuild?")
        if boolchoice==False:
            return
    print(f"Preprocessing {mainfilename}!")
    t1=process_time()
    pbar = tqdm(total=30)

    df=pd.read_csv(os.path.join(data_location,f"{mainfilename}.csv"))
    pbar.update(1)
    df2d=pd.read_csv(os.path.join(data_location,f"{metadatafilename}.csv"))

    pbar.update(1)
    current_objs = np.unique(df.object_id.values)
    all_objs = np.unique(df2d.object_id.values)

    indices_to_consider=np.intersect1d(current_objs,all_objs,return_indices=True)[2]
    df2d = df2d.iloc[indices_to_consider].reset_index(drop=True)
    pbar.update(1)
    df['flux']=np.random.normal(df['flux'], df['flux_err']/1.5)
    if "train" in mainfilename:
        flux_max = np.max(df['flux'])
        flux_min = np.min(df['flux'])
    else:
        #Lines added after training
        flux_max = 2751626.3243459687
        flux_min = -1596742.6487144697
    flux_pow = np.log2(flux_max - flux_min)
    df['flux']/=flux_pow
    df['flux_err']/=flux_pow
    pbar.update(1)
    # df['flux']/=pow(2,flux_pow)
    # df['flux_err']/=pow(2,flux_pow)

    train=df.copy()
    train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
    pbar.update(1)
    train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
    pbar.update(1)

    train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
    train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
    pbar.update(1)
    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std','sum','skew'],
        'flux_err': ['min', 'max', 'mean','skew'],
        'detected': [ 'mean', 'std','sum'],
        'flux_ratio_sq':['mean','sum','skew'],
        'flux_by_flux_ratio_sq':['mean','sum','skew'],
    }

    aggs_global = {
        'mjd': ['size'],
        'flux': ['min', 'max', 'mean', 'median', 'std','sum','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','sum','skew'],
        'detected': [ 'mean','skew','median','sum'],
        'flux_ratio_sq':['min', 'max', 'mean','sum','skew'],
        'flux_by_flux_ratio_sq':['min', 'max', 'mean','sum','skew'],
    }
    pbar.update(1)
    agg_train_global_feat = train.groupby('object_id').agg(aggs_global)

    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]

    new_columns_global = [
        k + '_' + agg for k in aggs_global.keys() for agg in aggs_global[k]
    ]

    agg_train_global_feat.columns = new_columns_global

    agg_train = train.groupby(['object_id','passband']).agg(aggs)

    agg_train = agg_train.unstack()
    pbar.update(1)
    col_names = []
    for col in new_columns:
        for i in range(6):
            col_names.append(col+'_'+str(i))

    agg_train.columns = col_names
    agg_train_global_feat['flux_diff'] = agg_train_global_feat['flux_max'] - agg_train_global_feat['flux_min']
    agg_train_global_feat['flux_dif2'] = (agg_train_global_feat['flux_max'] - agg_train_global_feat['flux_min']) / agg_train_global_feat['flux_mean']
    agg_train_global_feat['flux_w_mean'] = agg_train_global_feat['flux_by_flux_ratio_sq_sum'] / agg_train_global_feat['flux_ratio_sq_sum']
    agg_train_global_feat['flux_dif3'] = (agg_train_global_feat['flux_max'] - agg_train_global_feat['flux_min']) / agg_train_global_feat['flux_w_mean']
    pbar.update(1)
    # Legacy code. There are much better ways to compute this but for train set this suffices so 
    # i got too lazy to change https://www.kaggle.com/c/PLAsTiCC-2018/discussion/71398
    def detected_max(mjd,detected):
        try:     return max(mjd[detected==1]) - min(mjd[detected==1])
        except:  return 0

    temp = train.groupby('object_id').apply(lambda x:detected_max(x['mjd'],x['detected']))
    temp1 = train.groupby(['object_id','passband']).apply(lambda x:detected_max(x['mjd'],x['detected'])).unstack()
    temp.columns = ['mjd_global_diff']
    temp1.columns = ['mjd_pb0','mjd_pb1','mjd_pb2','mjd_pb3','mjd_pb4','mjd_pb5']
    temp = temp.reset_index()
    temp1 = temp1.reset_index()
    pbar.update(1)
    aggs_det = {
        'flux': ['min','mean', 'max','skew'],
        'flux_ratio_sq':['min','mean', 'max','skew'],
        'flux_by_flux_ratio_sq':['min', 'max','mean','skew'],
    }

    train_detected =  train[train.detected==1]
    temp2 = train_detected.groupby(['object_id']).agg(aggs_det)
    del(train_detected)
    del(train)
    new_columns_det = [
        k + '_det_' + agg for k in aggs_det.keys() for agg in aggs_det[k]
    ]

    temp2.columns = new_columns_det
    temp2['flux_diff_det'] = temp2['flux_det_max'] - temp2['flux_det_min']
    temp2['flux_ratio_sq_diff_det'] = temp2['flux_ratio_sq_det_max'] - temp2['flux_ratio_sq_det_min']
    temp2['flux_by_flux_ratio_sq_diff_det'] = temp2['flux_by_flux_ratio_sq_det_max'] - temp2['flux_by_flux_ratio_sq_det_min']
    pbar.update(1)
    del temp2['flux_by_flux_ratio_sq_det_max'],temp2['flux_by_flux_ratio_sq_det_min']
    del temp2['flux_ratio_sq_det_max'],temp2['flux_ratio_sq_det_min']
    del temp2['flux_det_max'],temp2['flux_det_min']

    meta_train = df2d.copy()
    del(df2d)
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
    meta_train["target"]=meta_train.loc[:,["target"]].replace(target_dict)

    full_train = agg_train.reset_index().merge(
        right=meta_train,
        how='outer',
        on='object_id'
    )
    del(agg_train)
    del(meta_train)
    full_train = full_train.merge(
        right=agg_train_global_feat,
        how='outer',
        on='object_id'
    )
    del(agg_train_global_feat)
    full_train = full_train.merge(
        right=temp,
        how='outer',
        on='object_id'
    )
    del(temp)
    full_train = full_train.merge(
        right=temp1,
        how='outer',
        on='object_id'
    )
    del(temp1)
    full_train = full_train.merge(
        right=temp2,
        how='outer',
        on='object_id'
    )
    del(temp2)
    labels = full_train['target']
    full_train.drop("target",axis=1,inplace=True)    
    pbar.update(1)

    if 'object_id' in full_train:
        oof_df = full_train[['object_id']]
        del full_train['object_id'], full_train['hostgal_specz']
        del full_train['ra'], full_train['decl'], full_train['gal_l'],full_train['gal_b'],full_train['ddf']

    useless_cols = [
    'flux_max_2',
    'flux_median_0',
    'flux_median_4',
    'flux_err_skew_1',
    'flux_err_skew_3',
    'detected_mean_4',
    'detected_std_3',
    'detected_std_4',
    'detected_sum_4',
    'flux_ratio_sq_mean_4',
    'flux_ratio_sq_sum_3',
    'flux_ratio_sq_sum_4',
    'flux_median',
    'flux_err_skew',
    'flux_ratio_sq_sum',
    'mjd_pb5',
    'flux_ratio_sq_det_skew',
    ]
    full_train_new = full_train.drop(useless_cols,axis=1)
    
    del(full_train)
    del(oof_df)

    from sklearn.preprocessing import PowerTransformer
    ss = PowerTransformer()
    twod_data = ss.fit_transform(np.nan_to_num(full_train_new))
    del(full_train_new)

    with open(os.path.join(pickle_location,f"{mainfilename}_2d_pickle"), "wb") as fp:   #Pickling
        pickle.dump(twod_data, fp)
    pbar.update(1)

    del(twod_data)


    #Calculate time diff between all observations in mjd
    df["mjd_diff"]=df['mjd'].diff()
    df["mjd_diff"]=df["mjd_diff"].fillna(0)

    #Find indexes where new objects appear, and set the mjd_diff for this to 0
    obj_change_index=np.where(df["object_id"].values[:-1] != df["object_id"].values[1:])[0] + 1
    df.loc[obj_change_index, ['mjd_diff']]=0

    # Use groupby method to find seperate cumsums for all objects
    df["cumulative_mjd_diff"]=df.loc[:,["object_id", "mjd_diff"]].groupby("object_id").cumsum()
    pbar.update(1)
    mjd_arr=df["mjd"].values
    time_diff_arr=df["mjd_diff"].values
    grouped_mjd_arr=np.zeros_like(mjd_arr)
    pbar.update(1)
    prev_time=0
    for i in range(len(mjd_arr)):
        current_time=mjd_arr[i]
        time_diff=time_diff_arr[i]
        if time_diff==0 or current_time-prev_time>0.33:
            grouped_mjd_arr[i]=current_time
            prev_time=current_time
        else:
            grouped_mjd_arr[i]=prev_time
    pbar.update(1)
    df["grouped_mjd"]=grouped_mjd_arr

    del(grouped_mjd_arr)
    del(time_diff_arr)
    del(mjd_arr)

    df = df.sort_values("flux_err").groupby(["object_id", "grouped_mjd", "passband"]).first()
    df = df.reset_index()
    pbar.update(1)
    #Drop all unnecessary columns. Note : mjd_diff and cumulative_mjd_diff are dropped as cause problems when pivoting. Will recalculate later
    pbar.update(1)
    df = df.drop(
        [
            "mjd",
            "detected",
            "mjd_diff",
            "cumulative_mjd_diff",
        ],
        axis=1,
    )

    mini_df=df[["object_id"]].groupby("object_id").first()

    df=df.drop(mini_df,axis=1)
    df = pd.pivot_table(df, index=["object_id","grouped_mjd"], columns=["passband"])
    df.columns= [f"{tup[0]}_passband_{tup[1]}" for tup in df.columns.values]
    df=df.reset_index(["grouped_mjd"])
    df=df.join(mini_df,how="left")
    pbar.update(1)
    del(mini_df)

    df=df.rename(columns={"grouped_mjd": "mjd"})
    df=df.reset_index()


    #Calculate time diff between all observations in mjd
    df["mjd_diff"]=df['mjd'].diff()
    df["mjd_diff"]=df["mjd_diff"].fillna(0)
    #Find indexes where new objects appear, and set the mjd_diff for this to 0
    obj_change_index=np.where(df["object_id"].values[:-1] != df["object_id"].values[1:])[0] + 1
    df.loc[obj_change_index, ['mjd_diff']]=0
    pbar.update(1)
    # Use groupby method to find seperate cumsums for all objects
    df["cumulative_mjd_diff"]=df.loc[:,["object_id", "mjd_diff"]].groupby("object_id").cumsum()
    pbar.update(1)
    df=df.set_index(["object_id"])

    df=df.drop(
        [
            "mjd",
        ],
        axis=1,
    )
    pbar.update(1)
    colstointer=["object_id","flux_passband_0","flux_passband_1","flux_passband_2","flux_passband_3","flux_passband_4","flux_passband_5","flux_err_passband_0","flux_err_passband_1","flux_err_passband_2","flux_err_passband_3","flux_err_passband_4","flux_err_passband_5","cumulative_mjd_diff"]
    df=df.reset_index()
    temp_df=df.loc[:,colstointer].groupby('object_id').apply(lambda group: group.set_index("cumulative_mjd_diff").interpolate(method="values").ffill().bfill())
    temp_df.reset_index(drop=True,inplace=True)
    df.loc[:,temp_df.columns]=temp_df
    del(temp_df)
    df.drop(["cumulative_mjd_diff"],1,inplace=True)
    df.set_index("object_id",inplace=True,drop=True)
    pbar.update(1)
    df = df.fillna(0)

    #Save some memory by converting float64 to float32 as 32 enough
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)

    pbar.update(1)
    #Recalculate time diff between all rows and set new objs to zero as before
    all_obj_ids=np.unique(df.index.get_level_values(0).values)
    dfarray=df.reset_index().to_numpy()
    all_obj_ids_long=dfarray[0:,0]
    all_labels_long=dfarray[0:,1]
    obj_change_index=np.where(all_obj_ids_long[:-1] != all_obj_ids_long[1:])[0] + 1

    pbar.update(1)
    tuplist=list(zip(np.insert(obj_change_index,0,0),obj_change_index))
    list_of_data_arrays=[]
    for tup in tuplist:
        list_of_data_arrays.append(dfarray[tup[0]:tup[1],1:])
    list_of_data_arrays.append(dfarray[obj_change_index[-1]:,1:])
    pbar.update(1)
    pbar.update(1)
    with open(os.path.join(pickle_location,f"{mainfilename}_3d_pickle"), "wb") as fp:   #Pickling
        pickle.dump(list_of_data_arrays, fp)
    pbar.update(1)
    with open(os.path.join(pickle_location,f"{mainfilename}_label_pickle"), "wb") as fp:   #Pickling
        pickle.dump(labels, fp)
    pbar.update(1)
    gc.collect()
    pbar.close()
    t2=process_time()
    print(f"Preprocessing took {t2-t1} seconds.")

def create_unblinded_set():
    url = "https://zenodo.org/record/2539456/files/plasticc_test_metadata.csv.gz?download=1"
    print("Downloading file...")
    r = requests.get(url, allow_redirects=True)
    open('plasticc_test_metadata.csv.gz', 'wb').write(r.content)
    print("Download complete.")
    print("Extracting file...")
    with gzip.open('./plasticc_test_metadata.csv.gz') as f:
        somedf = pd.read_csv(f)
    true_targetdf = somedf[["object_id","true_target"]]
    del(somedf)
    print("Extraction complete.")
    df = pd.read_csv(os.path.join(data_location,"test_set_metadata.csv"))
    df = df.merge(true_targetdf, on="object_id")
    df.to_csv(os.path.join(data_location,"unblinded_test_set_metadata.csv"), index=False)
    print(f"unblinded_test_set_metadata.csv has been saved successfully in {os.path.join(data_location,'unblinded_test_set_metadata.csv')}")
    
if __name__ == "__main__":
    n1="training_set"
    n2="training_set_metadata"
    csv_to_pickle(n1,n2)

if __name__ == "__main__":
    for i in range(1,12):
        n1=f"test_set_batch{i}"
        n2="unblinded_test_set_metadata"
        if not os.path.isfile(os.path.join(data_location,f"{n2}.csv")):
            unblindedchoice=query_yes_no("Preprocessing Test Data will require the true labels. This will require a 157 MB download. Do you want to proceed?")
            if unblindedchoice:
                create_unblinded_set()
        assert os.path.isfile(os.path.join(data_location,f"{n2}.csv"))
        csv_to_pickle(n1,n2)

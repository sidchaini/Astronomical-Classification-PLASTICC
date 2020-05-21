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

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".

    Original:https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
def get_input_directory():
    csv_folder=Path(input("Enter the folder containing Raw Data\n"))
    with open("config.txt","w") as config_file:
        config_file.write(str(csv_folder))
    return(csv_folder)
def get_pickle_directory():
    pickle_folder=Path(input("Enter the folder containing Pickle Data\n"))
    with open("config.txt","a") as config_file:
        config_file.write("\n")
        config_file.write(str(pickle_folder))
    return(pickle_folder)
def return_data_location():
    try:
        with open("config.txt") as config_file:
            print("Found config file! Loading it.")
            config_text = config_file.readlines()
            data_location=Path(config_text[0].strip())
    except FileNotFoundError:
        print("Config file not found! Creating one.")
        data_location=get_input_directory()
    print(data_location)
    while not os.path.isdir(data_location):
        print("Error in given Data folder. Resetting.")
        data_location=get_input_directory()
    print(f"Data Location = {data_location}")
    return (data_location)
def return_pickle_location():
    try:
        with open("config.txt") as config_file:
            config_text = config_file.readlines()
            pickle_location=Path(config_text[1])
        print("Found pickle location!")
    except IndexError:
        print("Pickle location not found in config")
        pickle_location=get_pickle_directory()
    while not os.path.isdir(data_location):
        print("Error in given Pickle folder. Resetting.")
        pickle_location=get_pickle_directory()
    print(f"Pickle Location = {pickle_location}")
    return pickle_location

def csv_to_pickle(mainfilename,metadatafilename):
    mainfilename=mainfilename.split(".csv")[0]
    metadatafilename=metadatafilename.split(".csv")[0]
    if os.path.isfile(fr"{pickle_location}\{mainfilename}_3d_pickle") or os.path.isfile(fr"{pickle_location}\{mainfilename}_2d_pickle") or os.path.isfile(fr"{pickle_location}\{mainfilename}_label_pickle"):
        boolchoice=query_yes_no(fr"{mainfilename}_pickle found! Do you want to rebuild?",default="no")
        if boolchoice==False:
            return
    print(f"Preprocessing {mainfilename}!")
    t1=process_time()
    pbar = tqdm(total=30)
    df=pd.read_csv(fr"{data_location}\{mainfilename}.csv")
    pbar.update(1)
    df2d=pd.read_csv(fr"{data_location}\{metadatafilename}.csv")
    pbar.update(1)
    if 'Unnamed: 0' in df2d.columns:
        df2d=df2d.drop('Unnamed: 0',1)

    current_objs = np.unique(df.object_id.values)
    all_objs = np.unique(df2d.object_id.values)

    indices_to_consider=np.intersect1d(current_objs,all_objs,return_indices=True)[2]
    df2d = df2d.iloc[indices_to_consider].reset_index(drop=True)
    pbar.update(1)
    df['flux']=np.random.normal(df['flux'], df['flux_err']/1.5)
    flux_max = np.max(df['flux'])
    flux_min = np.min(df['flux'])
    flux_pow = np.log2(flux_max - flux_min)
    
    df['flux']/=flux_pow
    df['flux_err']/=flux_pow
    pbar.update(1)
    # df['flux']/=pow(2,flux_pow)
    # df['flux_err']/=pow(2,flux_pow)

    if "true_target" in df2d.columns:
        df2d=df2d.rename(columns={"true_target": "target"})

    train=df.copy()
    train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
    pbar.update(1)
    train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
    pbar.update(1)
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'passband': ['min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_train = train.groupby('object_id').agg(aggs)
    pbar.update(1)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train.columns = new_columns
    pbar.update(1)
    agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
    agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
    pbar.update(1)
    agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
    agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
    pbar.update(1)
    agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']
    pbar.update(1)
    del agg_train['mjd_max'], agg_train['mjd_min']
    del train
    agg_train = agg_train.drop(["mjd_size","passband_min","passband_max","passband_mean","passband_median", "passband_std"],1)
    agg_train=agg_train.reset_index()
    pbar.update(1)

    df2d=df2d.merge(agg_train,on="object_id")
    del(agg_train)
    pbar.update(1)

    #Calculate time diff between all observations in mjd
    df["mjd_diff"]=df['mjd'].diff()
    df["mjd_diff"]=df["mjd_diff"].fillna(0)

    #Find indexes where new objects appear, and set the mjd_diff for this to 0
    obj_change_index=np.where(df["object_id"].values[:-1] != df["object_id"].values[1:])[0] + 1
    df.loc[obj_change_index, ['mjd_diff']]=0

    # Use groupby method to find seperate cumsums for all objects
    df["cumulative_mjd_diff"]=df.loc[:,["object_id", "mjd_diff"]].groupby("object_id").cumsum()
    pbar.update(1)
    #Use dictionary to create new column, replacing targets(6,15,...) by class(0,1,...)
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
        99: 14
    }
    df2d["target_class"]=df2d.loc[:,["target"]].replace(target_dict)

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
    df2d = df2d.drop(
        [
            "ra",
            "decl",
            "gal_b",
            "gal_l",
        ],
        axis=1,
    )
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
    df2d=df2d.set_index(["object_id"])

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

    #Save some memory by converting float64 to float32 as 32 enough
    for col in df2d.columns:
        if df2d[col].dtype == np.float64:
            df2d[col] = df2d[col].astype(np.float32)
    pbar.update(1)
    #Recalculate time diff between all rows and set new objs to zero as before
    all_obj_ids=np.unique(df.index.get_level_values(0).values)
    dfarray=df.reset_index().to_numpy()
    all_obj_ids_long=dfarray[0:,0]
    all_labels_long=dfarray[0:,1]
    obj_change_index=np.where(all_obj_ids_long[:-1] != all_obj_ids_long[1:])[0] + 1

    labels=df2d["target_class"].values
    pbar.update(1)
    tuplist=list(zip(np.insert(obj_change_index,0,0),obj_change_index))
    list_of_data_arrays=[]
    for tup in tuplist:
        list_of_data_arrays.append(dfarray[tup[0]:tup[1],1:])
    list_of_data_arrays.append(dfarray[obj_change_index[-1]:,1:])
    pbar.update(1)
    df2d = df2d.fillna(0)
    df2d.drop("target_class",axis=1,inplace=True)
    twod_data = df2d.reset_index().to_numpy()[:,1:]
    pbar.update(1)
    with open(fr"{pickle_location}\{mainfilename}_3d_pickle", "wb") as fp:   #Pickling
        pickle.dump(list_of_data_arrays, fp)
    pbar.update(1)
    with open(fr"{pickle_location}\{mainfilename}_2d_pickle", "wb") as fp:   #Pickling
        pickle.dump(twod_data, fp)
    pbar.update(1)
    with open(fr"{pickle_location}\{mainfilename}_label_pickle", "wb") as fp:   #Pickling
        pickle.dump(labels, fp)
    pbar.update(1)
    gc.collect()
    pbar.close()
    t2=process_time()
    print(f"Preprocessing took {t2-t1} seconds.")


data_location=return_data_location()
pickle_location=return_pickle_location()

if __name__ == "__main__":
    n1="training_set"
    n2="training_set_metadata"
    csv_to_pickle(n1,n2)

if __name__ == "__main__":
    for i in range(1,12):
        n1=f"test_set_batch{i}"
        n2="unblinded_test_set_metadata"
        csv_to_pickle(n1,n2)

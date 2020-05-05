#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
from time import process_time 
import pickle
import random
from pathlib import Path
import os
import sys
#from tqdm import tqdm
from tqdm.notebook import tqdm


# In[66]:


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


# In[67]:


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

data_location=return_data_location()
pickle_location=return_pickle_location()


# In[73]:


def csv_to_pickle(mainfilename,metadatafilename):
    mainfilename=mainfilename.split(".csv")[0]
    metadatafilename=metadatafilename.split(".csv")[0]
    if os.path.isfile(fr"{pickle_location}\{mainfilename}_pickle"):
        boolchoice=query_yes_no(fr"{mainfilename}_pickle found! Do you want to rebuild?")
        if boolchoice==False:
            return
    print(f"Preprocessing {mainfilename}!")
    t1=process_time()
    pbar = tqdm(total=100)
    df=pd.read_csv(fr"{data_location}\{mainfilename}.csv")
    pbar.update(100/32)
    df_metadata=pd.read_csv(fr"{data_location}\{metadatafilename}.csv")
    pbar.update(100/32)
    #Normalisation
    #Step1:
    #Comment and uncomment as required
    pbar.update(100/32)
    # all_flux=df.loc[:,"flux"].values
    # all_flux_err=df.loc[:,"flux_err"].values
    pbar.update(100/32)
    # all_hostgal_specz=df_metadata.loc[:,"hostgal_specz"].values
    # all_hostgal_photoz=df_metadata.loc[:,"hostgal_photoz"].values
    # all_hostgal_photoz_err=df_metadata.loc[:,"hostgal_photoz_err"].values
    pbar.update(100/32)
    #Step2:
    #Comment and uncomment as required

    # flux_mean=np.mean(all_flux)
    # flux_err_mean=np.mean(all_flux_err)
    # flux_std=np.std(all_flux)
    # flux_err_std=np.std(all_flux_err)
    pbar.update(100/32)
    # hostgal_specz_mean=np.mean(all_hostgal_specz)
    # hostgal_photoz_mean=np.mean(all_hostgal_photoz)
    # hostgal_photoz_err_mean=np.mean(all_hostgal_photoz_err)
    # hostgal_specz_std=np.std(all_hostgal_specz)
    # hostgal_photoz_std=np.std(all_hostgal_photoz)
    # hostgal_photoz_err_std=np.std(all_hostgal_photoz_err)
    pbar.update(100/32)
    #Step3:
    #Comment and uncomment as required
    pbar.update(100/32)
    # df.loc[:,"flux"]=(all_flux - flux_mean)/flux_std
    # df.loc[:,"flux_err"]=(all_flux_err - flux_err_mean)/flux_err_std
    pbar.update(100/32)
    # df_metadata.loc[:,"hostgal_specz"]=(all_hostgal_specz - hostgal_specz_mean)/hostgal_specz_std
    # df_metadata.loc[:,"hostgal_photoz"]=(all_hostgal_photoz - hostgal_photoz_mean)/hostgal_photoz_std
    # df_metadata.loc[:,"hostgal_photoz_err"]=(all_hostgal_photoz_err - hostgal_photoz_err_mean)/hostgal_photoz_err_std
    pbar.update(100/32)
    df=df.merge(df_metadata)
    pbar.update(100/32)
    del(df_metadata)
    if 'true_target' in df.columns:
        df=df.rename(columns={"true_target": "target"})
    #Comment and uncomment as required
    pbar.update(100/32)
    # del(all_flux)
    # del(all_flux_err)
    # del(all_hostgal_specz)
    # del(all_hostgal_photoz)
    # del(all_hostgal_photoz_err)
    pbar.update(100/32)
    #Calculate time diff between all observations in mjd
    df["mjd_diff"]=df['mjd'].diff()
    df["mjd_diff"]=df["mjd_diff"].fillna(0)
    pbar.update(100/32)
    #Find indexes where new objects appear, and set the mjd_diff for this to 0
    obj_change_index=np.where(df["object_id"].values[:-1] != df["object_id"].values[1:])[0] + 1
    df.loc[obj_change_index, ['mjd_diff']]=0
    pbar.update(100/32)
    # Use groupby method to find seperate cumsums for all objects
    df["cumulative_mjd_diff"]=df.loc[:,["object_id", "mjd_diff"]].groupby("object_id").cumsum()
    pbar.update(100/32)
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
        99: 14,
        991 : 14,
        992 : 14,
        993 : 14
    }
    df["target_class"]=df.loc[:,["target"]].replace(target_dict)
    pbar.update(100/32)
    mjd_arr=df["mjd"].values
    time_diff_arr=df["mjd_diff"].values
    grouped_mjd_arr=np.zeros_like(mjd_arr)
    pbar.update(100/32)
    prev_time=0
    for i in range(len(mjd_arr)):
        current_time=mjd_arr[i]
        time_diff=time_diff_arr[i]
        if time_diff==0 or current_time-prev_time>0.33:
            grouped_mjd_arr[i]=current_time
            prev_time=current_time
        else:
            grouped_mjd_arr[i]=prev_time
    pbar.update(100/32)
    df["grouped_mjd"]=grouped_mjd_arr
    del(grouped_mjd_arr)
    del(time_diff_arr)
    del(mjd_arr)
    pbar.update(100/32)
    df = df.sort_values("flux_err").groupby(["object_id", "grouped_mjd", "passband"]).first()
    df = df.reset_index()
    pbar.update(100/32)
    #Drop all unnecessary columns. Note : mjd_diff and cumulative_mjd_diff are dropped as cause problems when pivoting. Will recalculate later
    df = df.drop(
        [
            "mjd",
            "detected",
            "ra",
            "decl",
            "gal_b",
            "gal_l",
            "mjd_diff",
            "cumulative_mjd_diff",
    #         "ddf",                      #Experiment with these last 3
    #         "distmod",
    #         "mwebv"
        ],
        axis=1,
    )
    pbar.update(100/32)
    fixed_features=["ddf","hostgal_specz","hostgal_photoz","hostgal_photoz_err","distmod","mwebv"]
    mini_df=df[["object_id"] + fixed_features ].groupby("object_id").first()
    pbar.update(100/32)
    df=df.drop(mini_df,axis=1)
    df = pd.pivot_table(df, index=["object_id","grouped_mjd","target","target_class"], columns=["passband"])
    df.columns= [f"{tup[0]}_passband_{tup[1]}" for tup in df.columns.values]
    df=df.reset_index(["grouped_mjd","target","target_class"])
    df=df.join(mini_df,how="left")
    pbar.update(100/32)
    del(mini_df)

    df=df.rename(columns={"grouped_mjd": "mjd"})
    df=df.reset_index()
    pbar.update(100/32)

    #Calculate time diff between all observations in mjd
    df["mjd_diff"]=df['mjd'].diff()
    df["mjd_diff"]=df["mjd_diff"].fillna(0)
    pbar.update(100/32)
    #Find indexes where new objects appear, and set the mjd_diff for this to 0
    obj_change_index=np.where(df["object_id"].values[:-1] != df["object_id"].values[1:])[0] + 1
    df.loc[obj_change_index, ['mjd_diff']]=0
    df=df.set_index(["object_id"])
    df=df.drop(
        [
            "mjd",
            "target",
        ],
        axis=1,
    )
    df = df.fillna(0)
    #FUTURE: Try filling -1 and see if performance improves
    pbar.update(100/32)
    #Save some memory by converting float64 to float32 as 32 enough
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    pbar.update(100/32)
    #Recalculate time diff between all rows and set new objs to zero as before
    all_obj_ids=np.unique(df.index.get_level_values(0).values)
    dfarray=df.reset_index().to_numpy()
    all_obj_ids_long=dfarray[0:,0]
    all_labels_long=dfarray[0:,1]
    obj_change_index=np.where(all_obj_ids_long[:-1] != all_obj_ids_long[1:])[0] + 1
    pbar.update(100/32)
    tuplist=list(zip(np.insert(obj_change_index,0,0),obj_change_index))
    list_of_data_arrays=[]
    for tup in tuplist:
        list_of_data_arrays.append((dfarray[tup[0]:tup[1],2:],int(dfarray[tup[0],1]),int(dfarray[tup[0],0])))
    list_of_data_arrays.append((dfarray[obj_change_index[-1]:,2:],int(dfarray[obj_change_index[-1],1]),int(dfarray[obj_change_index[-1],0])))
    pbar.update(100/32)
    del(all_labels_long)
    del(all_obj_ids)
    del(all_obj_ids_long)
    del(df)
    del(dfarray)
    del(obj_change_index)
    del(tuplist)
    pbar.update(100/32)
    with open(fr"{pickle_location}\{mainfilename}_pickle", "wb") as fp:   #Pickling
        pickle.dump(list_of_data_arrays, fp)
    pbar.update(100/32)
    pbar.close()
    t2=process_time()
    print(f"Preprocessing took {t2-t1} seconds.")


# In[74]:


csv_to_pickle("training_set","training_set_metadata")


# In[75]:


csv_to_pickle("test_set_sample","test_set_sample_metadata")


# In[ ]:





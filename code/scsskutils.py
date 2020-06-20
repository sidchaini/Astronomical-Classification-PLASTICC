import pandas as pd
import numpy as np
from time import process_time 
import pickle
import random
from pathlib import Path
import os
import sys
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

data_location=return_data_location()
pickle_location=return_pickle_location()
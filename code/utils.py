"""
Utils
"""

import os
import csv

import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import pairwise_distance

def abs_path(local_filename, data_folder):
    """
    Gets the absolute path of the file given the name of the folder containing the data
    and the name of the file inside that folder, assuming that the repository contains a 
    'data' folder and a 'code' folder.

    :param local_filename: name of the data file
    :type local_filename: str
    :param data_folder: name of the folder which contains the data
    :type data_folder: str

    :return: the function returns the absolute path of the selected file
    :rtype: str
    
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # path of the code

    # Construct the absolute path to the data directory relative to the code directory
    data_dir = os.path.join(script_dir, "..", data_folder)

    # Construct the absolute path to the data file
    data_file_path = os.path.join(data_dir, local_filename)

    return data_file_path


def get_data(filepath, target_col=3, ex_cols=1, split = False):
    """
    Obtains the features and target arrays from a CSV file. Features and targets can be splitted in train and
    test through an optional argument.

    :param filepath: Path to the CSV file.
    :type filepath: str
    :param target_col: Number of last columns containing targets.
    :type target_col: (optional): int
    :param ex_cols: Number of initial columns to exclude from the features (default is 1).
    :type ex_cols: (optional): int
    :param split: I setted True the dataset is divided into train and test (default is False).
    :type split: (optional): Bool

    :return: NumPy arrays of features, targets.
    :rtype: tuple(numpy.ndarray, numpy.ndarray), if split = True tuple(numpy.ndarray, numpy.ndarray)

    """ 
    logger.info(f'Reading {os.path.basename(filepath)} with last {target_col} columns as targets')
    # Importing data from csv file as data
    data = pd.read_csv(filepath, delimiter = ',', header = None)
    # Excluding the first ex_cols columns
    data_df = data.iloc[:, ex_cols:]
    # Feature array (numpy.ndarray)
    features = data_df.iloc[:, :-target_col].values
    # Target array (numpy.ndarray)
    targets = data_df.iloc[:,-target_col:].values

    if len(features) != len(targets):
        logger.error("Number of samples in features and targets do not match ")
        raise ValueError("Mismatch between number of features and targets samples")

    if split:

        test_size = 0.1 # Percentage of test set 

        train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=test_size, random_state=40)

        logger.info(f"Training feature dataset with {train_features.shape} dimensions")
        logger.info(f"Training target dataset with {train_targets.shape} dimensions")
        logger.info(f"Testing feature dataset with {test_features.shape} dimensions")
        logger.info(f"Testing target dataset with {test_targets.shape} dimensions")

        return train_features, train_targets, test_features, test_targets
    
    else:
        logger.info(f'Feature dataset with {features.shape} dimension and Target dataset with {targets.shape} dimension')
        return features, targets


def get_outer(filepath, ex_cols=1):
    """
    A CSV reader that given the filepath and optionally the number of columns to be initially discarded, returns the values of the CSV

    :param filepath: Path to the CSV file.
    :type filepath: str
    :param ex_cols: Number of initial columns to exclude from the features (default is 1).
    :type ex_cols: (optional): int

    :return: NumPy darray of the dataset 
    :rtype: tuple(numpy.ndarray, numpy.ndarray), if split = True tuple(numpy.ndarray, numpy.ndarray)

    """ 
    logger.info(f'Reading {os.path.basename(filepath)}')
    # Importing data from csv file as data
    data = pd.read_csv(filepath, delimiter = ',', header = None)
    # Excluding the first ex_cols columns
    outer_data = data.iloc[:, ex_cols:].values

    return outer_data

def monk_data(data_file_path):

    '''
    This function takes as input the path to the document and splits it into columns. 
    Then it assigns the proper columns to the features and label variables X and y.

    Args:
        path: the path to the document
    '''

    data = pd.read_csv(data_file_path, header = None)
    df = data[0].str.split(expand=True)
    
    df.columns = ["target", "col_1", "col_2", "col_3", "col_4", "col_5", "col_6", "id"]
    
    y = df["target"]
    X = df.drop(columns=["target", "id"])

    return X, y

def standardize_data(train_features, test_features, train_targets, test_targets):
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    train_features = scaler_features.fit_transform(train_features)
    test_features = scaler_features.transform(test_features)
    train_targets = scaler_targets.fit_transform(train_targets)
    test_targets = scaler_targets.transform(test_targets)
    return train_features, test_features, train_targets, test_targets

def target_distribution(target, multitarget = False, show = False):
    """
    Given a multidimensional dataset it computes mean, standard deviation
    and range (max_value - min_value) for each column if multitarget is True, otherwise it computes the three
    values for the entire target dataset. Results are printed if show is setted as True

    :param target: the dataset
    :type target: numpy.ndarray
    :param multitarget: switch for the two computational options(default = False)
    :type multitarget: (optional) Bool
    :param show: if setted as True prints out the results(default = False)
    :type show: (optional) Bool

    :return: If multitarget is True it returns a np.matrix with this 
    structure [[mean], [std], [range]], where the three values are vectors;
    if multitarget is False it reurns a np.array of this kind [mean, std, range],
    where the three values are scalars
    :rtype: np.matrix or np.array
    """
    if multitarget:

        logger.info("Creating the matrix [[mean], [std], [range]] of the dataset")
        mean = target.mean(0)
        std = target.std(0)
        range = target.max(0)-target.min(0)

        result = np.matrix([mean, std, range])

        return 
    
    else:
        logger.info("Creating the vector [mean, std, range] of the dataset")
        mean = target.mean()
        std = target.std()
        range = target.max()-target.min()

        result = np.array([mean, std, range])
    
    #Selecting printing option
    if show:
        print(result)

    return result

# loss function for Keras and SVM models
def euclidean_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))

def euclidean_error_scorer(y_true, y_pred):
    mee = euclidean_error(y_true=y_true, y_pred=y_pred)
    return mee.numpy()

def euclidean_error_scorer2(y_true, y_pred):
    mee = euclidean_error(y_true=y_true, y_pred=y_pred)
    return mee.eval()

# it retrieves the mean value of all the passed losses
def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(euclidean_error(y_true, y_pred))

def mean_euclidean_error_scorer(y_true, y_pred):
    mee = mean_euclidean_error(y_true=y_true, y_pred=y_pred)
    return mee.numpy()

scorer = make_scorer(mean_euclidean_error_scorer, greater_is_better=False)

def w_csv(data):
    teamname = "DUNEDAIN"
    filename = f"{teamname}_ML-CUP24-TS.csv"

    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the custom header
            writer.writerow(["# Valerio Caporioni, Jacopo Resasco, Lavinia Rotellini"])
            writer.writerow([f"# {teamname}"])
            writer.writerow(["# ML-CUP24 V1"])
            writer.writerow(["# 03/01/2025"])

            for i, point in enumerate(data, start=1):
                if len(point) == 3:  # Ensure each point has 3 dimensions
                    # Write the id and dimensions in the same row
                    writer.writerow([i, point[0], point[1], point[2]])

        logger.info(f"CSV file '{filename}' written successfully!")
    except Exception as e:
        logger.error(f"Error writing CSV file")
        raise Exception 

def torch_mee(y_pred, y_true):
    return torch.mean(pairwise_distance(y_true, y_pred, p=2))

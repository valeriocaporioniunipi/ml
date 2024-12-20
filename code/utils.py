"""
Utils
"""

import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import make_scorer

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


def get_data(filename, target_col=3, ex_cols=1):
    """
    Obtains the features and target arrays from a CSV file.

    :param filename: Path to the CSV file.
    :type filename: str
    :param target_col: Number of last columns containing targets.
    :type target_col: (optional): int
    :param ex_cols: Number of initial columns to exclude from the features (default is 1).
    :type ex_cols: (optional): int

    :return: NumPy arrays of features, targets.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """ 
    logger.info(f'Reading {os.path.basename(filename)} with last {target_col} columns as targets')
    # Importing data from csv file as data
    data = pd.read_csv(filename, delimiter = ',', header = None)
    # Excluding the first ex_cols columns
    data_df = data.iloc[:, ex_cols:]
    # Feature array (numpy.ndarray)
    features = data_df.iloc[:, :-target_col].values
    # Target array (numpy.ndarray)
    targets = data_df.iloc[:,-target_col:].values

    logger.info(f'Feature dataset with {features.shape} dimension and Target dataset with {targets.shape} dimension')

    if len(features) != len(targets):
        logger.error("Number of samples in features and targets do not match ")
        raise ValueError("Mismatch between number of features and targets samples")
    return features, targets


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
    return np.sqrt(np.sum(np.square(y_pred - y_true), axis=1))


# it retrieves the mean value of all the passed losses
def mean_euclidean_error(y_true, y_pred):
    return np.mean(euclidean_error(y_true, y_pred))


def scorer():
    make_scorer(mean_euclidean_error, greater_is_better=False)
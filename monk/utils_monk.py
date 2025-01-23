### SCRIPT DEFINING USEFUL FUNCTIONS USED IN MONKS
import os 
import pandas as pd

# Function to divide the dataset into columns
def read_split_data(data_file_path):

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
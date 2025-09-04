import os
import sys

import numpy as np
import pandas as pd
import dill  # For serializing Python objects
import pickle  # Another option for serialization
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.

    Parameters:
    -----------
    file_path : str
        The full path where the object will be saved (including filename).
    obj : any Python object
        The Python object to serialize and save.

    Raises:
    -------
    CustomException
        If there is an error during the saving process.
    """
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Use dill to serialize the object

    except Exception as e:
        # Raise a custom exception with system info
        raise CustomException(e, sys)

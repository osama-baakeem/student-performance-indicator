import os
import sys

import numpy as np
import pandas as pd
import dill  # For serializing Python objects (supports more object types than pickle)
import pickle  # Another option for serialization (standard library)
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
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and serialize the object with dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # dill can handle a wider range of objects than pickle

    except Exception as e:
        # Wrap and raise the exception as a CustomException for better debugging
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple models with hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    X_train : array-like
        Training feature set.
    y_train : array-like
        Training labels.
    X_test : array-like
        Testing feature set.
    y_test : array-like
        Testing labels.
    models : dict
        Dictionary containing model names as keys and model objects as values.
    param : dict
        Dictionary containing model names as keys and parameter grids for GridSearchCV as values.

    Returns:
    --------
    report : dict
        Dictionary with model names as keys and their test R² scores as values.
    fitted_models : dict
        Dictionary with model names as keys and their best-fitted model objects as values.
    """
    try:
        report = {}         # Store evaluation results
        fitted_models = {}  # Store best-fitted models

        # Loop through each model in the provided dictionary
        for i, (model_name, model) in enumerate(models.items()):
            # Fetch the corresponding hyperparameter grid for the current model
            para = param[model_name]

            # Perform hyperparameter tuning using GridSearchCV with 3-fold cross-validation
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Initialize a fresh model instance with the best found hyperparameters
            best_model = type(model)(**gs.best_params_)
            best_model.fit(X_train, y_train)

            # Generate predictions for training and testing sets
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R² scores for training and testing sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test performance and fitted model
            report[model_name] = test_model_score
            fitted_models[model_name] = best_model   

        return report, fitted_models   

    except Exception as e:
        # Wrap any error as a CustomException
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a serialized Python object from a file using pickle.

    Parameters:
    -----------
    file_path : str
        The full path of the file containing the serialized object.

    Returns:
    --------
    obj : any Python object
        The deserialized Python object.

    Raises:
    -------
    CustomException
        If there is an error during the loading process.
    """
    try:
        # Open the file in binary read mode and deserialize the object with dill
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        # Wrap and raise the exception as a CustomException
        raise CustomException(e, sys)

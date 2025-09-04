import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Configuration class to store file paths or other settings
@dataclass
class DataTransformationConfig:
    # Path to save the preprocessor object
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


# Main class for data transformation
class DataTransformation:
    def __init__(self):
        # Initialize the configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates and returns a preprocessing object
        that applies transformations to numerical and categorical features.
        """
        try:
            # Define which columns are numerical and which are categorical
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features: fill missing values + scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # replace NaNs with median
                    ("scaler", StandardScaler())  # standardize to mean=0, std=1
                ]
            )

            # Pipeline for categorical features: fill missing values + encode + scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill NaNs with most frequent
                    ("one_hot_encoder", OneHotEncoder()),  # convert categories to dummy variables
                    ("scaler", StandardScaler(with_mean=False))  # scale without centering for sparse matrix
                ]
            )

            # Log columns info
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine numerical and categorical pipelines into one preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function reads training and testing data, applies preprocessing,
        and returns transformed arrays along with the path of the saved preprocessor.
        """
        try:
            # Read train and test CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Get the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define target variable and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing pipelines
            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine preprocessed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object for future use
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
        

        
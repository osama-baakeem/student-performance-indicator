import sys
import os
import pandas as pd
from src.exception import CustomException  # Custom exception handler for better error tracking
from src.utils import load_object          # Utility function to load serialized objects (e.g., pickle files)


# Prediction Pipeline Class
class PredictPipeline:
    def __init__(self):
        # Currently no initialization logic is needed
        pass

    def predict(self, features):
        """
        Perform prediction using the trained model and preprocessor.

        Args:
            features (DataFrame): Input features provided by the user (structured data).
        
        Returns:
            preds (array): Model predictions.
        """
        try:
            # Define paths to saved model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")  # Debugging checkpoint

            # Load trained model and preprocessor objects from artifacts
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")  # Debugging checkpoint

            # Apply preprocessing transformations to input features
            data_scaled = preprocessor.transform(features)

            # Generate predictions using the trained model
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            # Raise custom exception for debugging/logging
            raise CustomException(e, sys)



# Custom Data Class
class CustomData:
    """
    A helper class to structure raw user input into a DataFrame,
    so that it can be used with the trained ML pipeline.
    """

    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        # Store user input as class attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert user input into a Pandas DataFrame.
        
        Returns:
            DataFrame: Structured input ready for preprocessing and prediction.
        """
        try:
            # Create a dictionary mapping column names to user input values
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to DataFrame (one row of input data)
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

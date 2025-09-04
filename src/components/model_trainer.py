import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainConfig:
    """
    Configuration class for storing model training parameters.

    Attributes:
    -----------
    trained_model_file_path : str
        Path to save the trained model.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Handles the training, evaluation, and saving of regression models.
    """

    def __init__(self):
        # Initialize configuration
        self.model_trainer_congif = ModelTrainConfig

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple models with hyperparameter tuning, 
        evaluate their performance, and save the best model.

        Parameters:
        -----------
        train_array : numpy.ndarray
            Combined training data (features + target in the last column).
        test_array : numpy.ndarray
            Combined testing data (features + target in the last column).

        Returns:
        --------
        r2_value : float
            R² score of the best model on the test dataset.

        Raises:
        -------
        CustomException
            If no suitable model is found or an error occurs.
        """
        try:
            logging.info("Splitting train and test input data")

            # Split features (X) and target (y) from train and test arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # Features for training
                train_array[:, -1],    # Target for training
                test_array[:, :-1],    # Features for testing
                test_array[:, -1]      # Target for testing
            )

            # Dictionary of regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grids for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparameters for tuning
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Train and evaluate models using utility function
            logging.info("Starting model evaluation using GridSearchCV")
            model_report, fitted_models = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Find the best model based on test R² score
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = fitted_models[best_model_name]

            if best_model_score < 0.6:
                # Raise exception if no model performs well enough
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with R2 score {best_model_score}")

            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_congif.trained_model_file_path,
                obj=best_model
            )

            # Evaluate best model on the test set
            predicted = best_model.predict(X_test)
            r2_value = r2_score(y_test, predicted)

            return r2_value

        except Exception as e:
            raise CustomException(e, sys)

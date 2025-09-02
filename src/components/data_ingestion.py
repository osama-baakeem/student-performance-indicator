# Import libraries
import os  # For file path operations and directory management
import sys  # For accessing system-specific parameters and exception info
import pandas as pd  # For data manipulation and CSV handling

# Import custom modules for logging and exception handling
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Custom logging configuration

# Import scikit-learn function for splitting data
from sklearn.model_selection import train_test_split

# Import dataclass decorator for configuration classes
from dataclasses import dataclass

# Import custom modules for data transformation and model training
#from src.components.data_transformation import DataTransformation, DataTransformationConfig
#from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

# =========================
# Configuration for Data Ingestion
# =========================

@dataclass
class DataIngestionConfig:
    """
    Configuration class for paths of raw, train, and test data.
    """
    base_dir: str = os.getcwd()  # project root (where script is run)
    train_data_path: str = os.path.join(base_dir, 'artifacts', "train.csv")  # Location to save training data.
    test_data_path: str = os.path.join(base_dir, 'artifacts', "test.csv")  # Location to save testing data.
    raw_data_path: str = os.path.join(base_dir, 'artifacts', "data.csv")  # Location to save raw data.



# =========================
# Data Ingestion Class
# =========================
class DataIngestion:
    """
    Handles reading raw data, splitting into train/test, and saving CSVs.
    """
    def __init__(self):
        # Initialize configuration for file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads dataset, splits it into training and test sets, saves CSVs, and returns paths.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv') 
            logging.info('Read the dataset as DataFrame')

            # Ensure the artifacts directory exists, if not create it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data for record-keeping
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Split data into training and test sets
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved successfully")

            # Return the paths of saved CSV files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Raise custom exception in case of error
            raise CustomException(e, sys)

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # Step 1: Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
























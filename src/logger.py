import logging
import os
from datetime import datetime

# Generate a unique log file name using current date & time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory (all log files will be stored here)
logs_dir = os.path.join(os.getcwd(), "logs", LOG_FILE)  # getcwd: get current working directory

# Create the logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Log messages will be written here
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  
    # Format of each log line:
    # %(asctime)s  -> timestamp of log
    # %(lineno)d   -> line number in code
    # %(name)s     -> logger name (default: root)
    # %(levelname)s-> log level (INFO, ERROR, etc.)
    # %(message)s  -> actual log message
    level=logging.INFO,  # Minimum logging level (INFO and above will be logged)
)




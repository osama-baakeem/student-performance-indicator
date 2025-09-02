import sys
from src.logger import logging


# Function to create a detailed error message with file name, line number, and the actual error message
def error_message_detail(error, error_detail: sys):
    """
    error: The actual error/exception object.
    error_detail: Usually 'sys' module (used to fetch exception details).
    """
    # Extract the exception information (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()

    # Get the name of the file where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format a detailed error message with file name, line number, and the error itself
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,  # the script file name
        exc_tb.tb_lineno,  # the line number where the error occurred
        str(error)  # string representation of the error message
    )

    return error_message

# Custom Exception class for better error reporting
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """ 
        error_message: The original error message
        error_detail: The sys module to fetch traceback details
        """
        # Call the parent class (Exception) constructor
        super().__init__(error_message)

        # Create a detailed error message using our helper function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        # When the exception is printed, return the detailed message
        return self.error_message




'''
Benefits of this approach:
-You control the error format → can log it in a structured way.
-Useful in large applications (e.g., machine learning pipelines, web apps, microservices) where:
    -You want consistent logging
    -You may save errors to a file/DB/monitoring system
    -You do not want the full Python traceback but a clean custom format

=> For production-level applications → yes, CustomException (or similar) is useful to standardize
   error messages and integrate with logging systems.
'''

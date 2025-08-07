import sys
from exception_logging.logger import logging

def error_message_details(error, error_details:sys):
    _,_, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name} at line number: {exc_tb.tb_lineno} with error message: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_details:sys):
        super().__init__(error)
        self.error_message = error_message_details(error, error_details)

    def __str__(self):
        return self.error_message
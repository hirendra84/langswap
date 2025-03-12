import os
import json
from typing import List
import logging

class Logger:
    def __init__(self, directory) -> None:
        self.directory = directory

        self.file_logger = self.setup_logger()

    def setup_logger(self, log_file='logs.log'):
        logger = logging.getLogger('file_logger')
        logger.setLevel(logging.DEBUG)

        log_file = os.path.join(self.directory, log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def log_json(self, file_name, data: List):
        log_text = os.path.join(self.directory, file_name)
        with open(log_text, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
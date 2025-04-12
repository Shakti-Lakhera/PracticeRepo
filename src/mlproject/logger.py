import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #Create file
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)  #Create file path
os.makedirs(log_path,exist_ok=True)  #Check directory

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)  #Log actual file path

#Task of logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
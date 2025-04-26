import logging
import os
from datetime import datetime

LOG__FILE =f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path =os.path.join(os.getcwd(),"logs" ,LOG__FILE)
os.makedirs(logs_path, exist_ok=True) 

LOG__FILE_PATH = os.path.join(logs_path, LOG__FILE)

logging.basicConfig(
    filename=LOG__FILE_PATH, 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
import logging
import datetime
import os

root = os.path.dirname(os.path.dirname(__file__))
log_folder = os.path.join(root, 'log')

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

curr_date = datetime.datetime.today()
formated_curr_date = curr_date.strftime("%Y-%m-%d-%H.%M.%S")

# format="%(asctime)s - %(name)s - %(levelname)s : %(message)s"
# corresponds to
# <current date> - <module name> - <level name> : <message> 
# example: 2021-03-04 10:42:16,zzz - __main__ - INFO : Loading pretrained NLTK model

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s : %(message)s",
                    level=logging.INFO,
                    filename=f"{log_folder}/surveynlp_{formated_curr_date}.log",
                    filemode="w")

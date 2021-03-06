import os
import sys
sys.path.append(os.path.dirname(__file__))

import logging
import logger_init

logger = logging.getLogger(__name__)

def evaluate_path_exists(model_dir):
    msg = f"{model_dir} not found!"

    if os.path.exists(model_dir):
        logger.info(f"{model_dir} found!")
        return None
    
    logger.error(msg)
    sys.exit(msg)

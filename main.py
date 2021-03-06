import os
import sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import logging
import sentiment.logger_init

from sentiment.models import Sentiment_models

logger = logging.getLogger(__name__)

if __name__=="__main__":
    curr_dir = os.path.dirname(__file__)
    filename = "data/sentiment140.csv"

    logger.info(f"Reading file {filename}")
    df = pd.read_csv(os.path.join(curr_dir, filename), error_bad_lines=False, sep=';')

    verbatims = df.text.to_list()
    S = Sentiment_models(verbatims)
    sentiments, scores = S.get_predictions()

    logger.info("All done!")

    print(sentiments)
    print(scores)

    print(pd.crosstab(df.polarity, sentiments["majority vote"]))
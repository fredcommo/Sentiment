import os
import sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import logging
from sentiment.logger import logger_init

from sentiment.models.sentiment_models import Sentiment_models

logger = logging.getLogger(__name__)

def crosstab(series1, series2):
    ct = pd.crosstab(series1, series2)
    props = ct.apply(lambda row: row/row.sum(), axis=1)
    return props

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

    for serie in sentiments.columns:
        print(crosstab(df.label, sentiments[serie]))
        print()

    # Majority vote crosstable
    print(pd.crosstab(df.label, sentiments["majority vote"]))
    print()

    # Weighted vote crosstable
    print(pd.crosstab(df.label, sentiments["weighted vote"]))
    print()

    # Voting methods performances (accuracy)
    acc_maj_vote = np.mean(df.label == sentiments["majority vote"])
    acc_weighted_vote = np.mean(df.label == sentiments["weighted vote"])
    print("Majority vote accuracy: {:.3f}".format(acc_maj_vote))
    print("Weighted vote accuracy: {:.3f}".format(acc_weighted_vote))
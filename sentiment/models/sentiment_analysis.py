import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(__file__))

import logging
from sentiment.logger import logger_init
from sentiment.models.pretrained_sentiment_models import *

import pickle

logger = logging.getLogger(__name__)

DISTILBERT_SENT = "distilbert-base-uncased-finetuned-sst-2-english"
ROBERTA_SENT = "VictorSanh/roberta-base-finetuned-yelp-polarity"
TWITTER_SENT = "cardiffnlp/twitter-roberta-base-sentiment"
BERT_SENT = "nlptown/bert-base-multilingual-uncased-sentiment"

# TAPAS_SENT not working: requires specific data format, not documented
# url = "google/tapas-base-finetuned-tabfact"

class Sentiments(object):
    
    def __init__(self, verbatims):

        logger.info("running TEXTBLOB sentiment model")
        self.txtblob_sent = map(self._sentiment_textblob, verbatims)
        
        logger.info("running FLAIR sentiment model")
        self.flair = map(self._sentiment_flair, verbatims)
        
        logger.info("running DISTILBERT sentiment model")
        self.distilbert = self._transformer_classification(verbatims, DISTILBERT_SENT)
        
        logger.info("running ROBERTA sentiment model")
        self.roberta = self._transformer_classification(verbatims, ROBERTA_SENT)

        logger.info("running TWITTER-ROBERTA sentiment model")
        self.twitter = self._transformer_classification(verbatims, TWITTER_SENT)

        logger.info("running BER sentiment model")
        self.bert = self._transformer_classification(verbatims, BERT_SENT)    
    
    
    @staticmethod
    def _sentiment_nltk(text):
        return sentiment_nltk(text)

    @staticmethod
    def _sentiment_textblob(text):
        return sentiment_textblob(text)

    @staticmethod
    def _sentiment_flair(text):
        return sentiment_flair(text)

    @staticmethod
    def _transformer_classification(verbatims, hugging_face_model):
        return transformer_classification(verbatims, hugging_face_model)
    
    @staticmethod
    def majority_vote(series):

        def _neutral_is_in_index(counts):
            return "neutral" in counts[counts == counts.max()]

        def _both_neg_and_pos(counts):
            top_count = counts[counts == counts.max()]
            return 'negative' in top_count and 'positive' in top_count

        def _vote(counts):
            if _neutral_is_in_index(counts):
                return "neutral"
            if _both_neg_and_pos(counts):
                return "neutral"
            return str(counts[counts == counts.max()].index[0])

        if series.isnull().all():
            return None

        return _vote(series.value_counts())

    @staticmethod
    def weighted_vote(series):
        
        def _load_pickle(filename):
            try:
                filepath = os.path.dirname(__file__) + f"/weighted_vote/{filename}.sav"
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"{filename}.sav loaded")
                return model
            except Exception as e:
                msg = f"{filename}.sav not loaded"
                logger.exception(msg, exc_info=True)
                sys.exit(msg)

        encoder = _load_pickle("one_hot_encoder")
        logreg = _load_pickle("weighted_vote")

        features = encoder.transform(series).toarray()
        weighted_vote = logreg.predict(features)
        
        return weighted_vote

    def get_predictions(self):
        try:
            logger.info("Extracting all sentiment predictions.")
            list_predictions = [pd.DataFrame(sent) for sent in self.__dict__.values()]

            logger.info("Concatenating all sentiment predictions.")
            predictions = pd.concat(list_predictions, axis=1)
            label_columns = predictions.columns.str.contains('label')

            try:
                logger.info("Extracting all sentiment predictions.")
                sentiments = predictions.loc[:,label_columns]
            except Exception as e:
                logger.exception("label colmnns not found", exc_info=True)

            try:
                logger.info("Running naive majority vote.")
                majority_vote = sentiments.apply(self.majority_vote, axis=1)
            except Exception as e:
                logger.exception("Error when majority voting", exc_info=True)

            try:
                logger.info("Running weighted vote.")
                weighted_vote = self.weighted_vote(sentiments)
            except Exception as e:
                logger.exception("Error when weighted voting", exc_info=True)

            sentiments['majority vote'] = majority_vote
            sentiments['weighted vote'] = weighted_vote

            try:
                probabilities = predictions.loc[:,~label_columns]
                return sentiments, probabilities
            except Exception as e:
                logger.exception("Error when getting probability columns", exc_info=True)

        except Exception as e:
            logger.exception("An error occured when getting models predictions", exc_info=True)

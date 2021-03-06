import os
import sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import flair
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging
import logger_init
from models_utils import evaluate_path_exists

from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


def sentiment_nltk(text):

    def _standardize_sent_labels(original_label):
        label_mapping = {"pos": "positive", "neu": "neutral", "neg": "negative"}
        return label_mapping[original_label]

    if not text:
        return {"nltk_label": None, "nltk_score": np.nan}

    sid = SentimentIntensityAnalyzer()
    nltk_scores = sid.polarity_scores(text)
    nltk_scores.pop("compound", None)
    sent = sorted(nltk_scores, key=nltk_scores.get, reverse=True)[0]
    return {"nltk_label": _standardize_sent_labels(sent), "nltk_score": nltk_scores[sent]}


def sentiment_textblob(text):

    def _polarity_to_sentiment(polarity):
        if polarity > 0.3:
            return 'positive'
        elif polarity < -0.3:
            return 'negative'
        else:
            return 'neutral'

    if not text:
        return {"textblob_label": None, "textblob_score": np.nan}
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return {"textblob_label": _polarity_to_sentiment(polarity), "textblob_score": polarity}


def sentiment_flair(text):

    if not text:
        return {'flair_label': None, "flair_score": np.nan}

    sent = flair.data.Sentence(text)
    flair_model.predict(sent)
    sentiment_dict = sent.labels[0].to_dict()
    return {'flair_label': sentiment_dict["value"].lower(), "flair_score": sentiment_dict['confidence']}


def transformer_classification(verbatims, model_dir):

    def _initialize_classifier(model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classifier

    def _get_batch_value(text, batch, model_name):
        if not text:
            return {f'{model_name}_label': None, f'{model_name}_score': np.nan}
        return {f'{model_name}_label': batch[0]['label'], f'{model_name}_score': batch[0]['score']}

    model_name = os.path.basename(model_dir).split('_')[1]
    classifier = _initialize_classifier(model_dir)
    batches = zip(verbatims, map(classifier, verbatims))
    return [_get_batch_value(text, batch, model_name) for text, batch in batches]


class Sentiment_models(object):
    
    def __init__(self, verbatims):
        logger.info("running NLTK sentiment model")
        self.nltk_sent = map(self._sentiment_nltk, verbatims)

        logger.info("running TEXTBLOB sentiment model")
        self.txtblob_sent = map(self._sentiment_textblob, verbatims)
        
        logger.info("running FLAIR sentiment model")
        self.flair = map(self._sentiment_flair, verbatims)
        
        logger.info("running DISTILBERT sentiment model")
        self.distilbert_sent = self._transformer_classification(verbatims, DISTILBERT_SENT)
        
        logger.info("running ROBERTA sentiment model")
        self.roberta_sent = self._transformer_classification(verbatims, ROBERTA_SENT)
    
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
    def _transformer_classification(verbatims, model_dir):
        return transformer_classification(verbatims, model_dir)
    
    @staticmethod
    def sentiment_vote(series):
        if series.isnull().all():
            return None
        counts = series.value_counts()
        max_count = counts[0]
        return "/".join(counts[counts == max_count].index)

    def get_predictions(self):
        try:
            logger.info("Compiling all sentiment predictions.")
            list_predictions = [pd.DataFrame(sent) for sent in self.__dict__.values()]
            predictions = pd.concat(list_predictions, axis=1)
            label_columns = predictions.columns.str.contains('label')
            sentiments = predictions.loc[:,label_columns]
            sentiments['majority vote'] = sentiments.apply(self.sentiment_vote, axis=1)
            scores = predictions.loc[:,~label_columns]
            return sentiments, scores
        except Exception as e:
            logger.exception("Error occured when getting models predictions", exc_info=True)

logger.info("Loading pretrained models...")

try:
    nltk.download("vader_lexicon")
except Exception as e:
    msg = 'The program failed to load NLTK'
    logger.exception(msg, exc_info=True)
    sys.exit(msg)

try:
    flair_model = flair.models.TextClassifier.load("en-sentiment")
except Exception as e:
    msg = 'The program failed to load FLAIR'
    logger.exception(msg, exc_info=True)
    sys.exit(msg)

root = os.path.dirname(os.path.dirname(__file__))
DISTILBERT_SENT = os.path.join(root, 'pretrained_models/sentiment_distilbert')
ROBERTA_SENT = os.path.join(root, 'pretrained_models/sentiment_roberta')

evaluate_path_exists(DISTILBERT_SENT)
evaluate_path_exists(ROBERTA_SENT)

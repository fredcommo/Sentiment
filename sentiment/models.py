import os
import sys
sys.path.append(os.path.dirname(__file__))

import re
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


logger.info("Loading pretrained models...")

try:
    nltk.download("vader_lexicon")
    logger.info("NLTK loaded")
except Exception as e:
    msg = 'The program failed to load NLTK'
    logger.exception(msg, exc_info=True)
    sys.exit(msg)

try:
    flair_model = flair.models.TextClassifier.load("en-sentiment")
    logger.info("FLAIR loaded")
except Exception as e:
    msg = 'The program failed to load FLAIR'
    logger.exception(msg, exc_info=True)
    sys.exit(msg)

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


def transformer_classification(verbatims, hugging_face_model):

    def _get_simplified_model_name(hugging_face_model):
        return os.path.basename(hugging_face_model).split('-')[0]

    def _initialize_classifier(hugging_face_model):
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
        model = AutoModelForSequenceClassification.from_pretrained(hugging_face_model)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return classifier

    def _standardize_label(model_name, label):
        
        roberta_corresp = {"LABEL_0": "negative", "LABEL_1": "positive"}
        twitter_corresp = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        bert_corresp = {
            "1 star": "negative",
            "2 stars": "neutral",
            "3 stars": "neutral",
            "4 stars": "neutral",
            "5 stars": "positive"
            }
        
        # label formats returned by ROBERTA and TWITTER-ROBERTA
        if re.match("LABEL_\\d", label):
            try:
                if re.search('roberta', model_name):
                    label = roberta_corresp[label]
                elif re.search('twitter', model_name):
                    label = twitter_corresp[label]
            except:
                logger.error(f"{label} found in {model_name} not in standardized dict")
        
        # label formats returned by BERT 
        if re.search("star", label):
            try:
                label = bert_corresp[label]
            except:
                logger.error(f"{label} found in {model_name} not in standardized dict")

        return label.lower()

    def _get_batch_value(text, batch, model_name):
        if not text:
            return {f'{model_name}_label': None, f'{model_name}_score': np.nan}

        label = _standardize_label(model_name, batch[0]['label'])
        return {f'{model_name}_label': label, f'{model_name}_score': batch[0]['score']}

    model_name = _get_simplified_model_name(hugging_face_model)
    
    logger.info(f"Initializing {model_name}")
    classifier = _initialize_classifier(hugging_face_model)

    batches = zip(verbatims, map(classifier, verbatims))
    return [_get_batch_value(text, batch, model_name) for text, batch in batches]

DISTILBERT_SENT = "distilbert-base-uncased-finetuned-sst-2-english"
ROBERTA_SENT = "VictorSanh/roberta-base-finetuned-yelp-polarity"
TWITTER_SENT = "cardiffnlp/twitter-roberta-base-sentiment"
BERT_SENT = "nlptown/bert-base-multilingual-uncased-sentiment"
TAPAS_SENT = "google/tapas-base-finetuned-tabfact"

class Sentiment_models(object):
    
    def __init__(self, verbatims):
        # logger.info("running NLTK sentiment model")
        # self.nltk_sent = map(self._sentiment_nltk, verbatims)

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

        logger.info("running GOOGLE-TAPAS sentiment model")
        self.tapas = self._transformer_classification(verbatims, TAPAS_SENT)    
    
    
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
    def sentiment_vote(series):

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

        # return "/".join(counts[counts == max_count].index)

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
                sentiments['majority vote'] = sentiments.apply(self.sentiment_vote, axis=1)
            except Exception as e:
                logger.exception("Error when voting", exc_info=True)

            try:
                scores = predictions.loc[:,~label_columns]
                return sentiments, scores
            except Exception as e:
                logger.exception("Error when getting score columns", exc_info=True)

        except Exception as e:
            logger.exception("An error occured when getting models predictions", exc_info=True)

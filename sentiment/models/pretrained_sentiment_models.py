
import re
import os
import numpy as np

import flair
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import logging
from sentiment.logger import logger_init

logger = logging.getLogger(__name__)


logger.info("Loading pretrained sentiment models...")

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

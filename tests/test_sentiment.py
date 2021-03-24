import pytest
import numpy as np
from sentiment.models.sentiment_models import sentiment_textblob


class Test_Sentiment_Textblob():

    def __init__(self):
        self.valid_verbatim = "I love python very much"
        self.empty_verbatim = ""
        self.special_char = "...$*^!:)"

    def test_sentiment_textblob_valid_verbatim(self):
        verbatim = self.valid_verbatim
        output = sentiment_textblob(verbatim)
        
        assert isinstance(output, dict)
        assert "textblob_label" in output.keys(), "Key 'textblob_label' not in output"
        assert "textblob_score" in output.keys(), "Key 'textblob_score' not in output"
        
        msg = f"{verbatim} label was expected 'positive', but got {output['textblob_label']}"
        assert output["textblob_label"] == "positive", msg

        msg = f"{verbatim} score was expected > 0, but got {output['textblob_score']}"
        assert output["textblob_score"] > 0, msg

    def test_sentiment_textblob_empty_verbatim(self):
        verbatim = self.empty_verbatim
        expected = {"textblob_label": None, "textblob_score": np.nan}
        output = sentiment_textblob(verbatim)

        assert isinstance(output, dict)
        assert "textblob_label" in output.keys(), "Key 'textblob_label' not in output"
        assert "textblob_score" in output.keys(), "Key 'textblob_score' not in output"
        
        msg = f"{verbatim} label was expected as None, but got {output['textblob_label']}"
        assert output == expected, msg

    def test_sentiment_textblob_verbatim_with_special_char(self):
        verbatim = self.special_char
        output = sentiment_textblob(verbatim)

        assert isinstance(output, dict)
        assert "textblob_label" in output.keys(), "Key 'textblob_label' not in output"
        assert "textblob_score" in output.keys(), "Key 'textblob_score' not in output"
        
        msg = f"{verbatim} label was expected as not None, but got {output['textblob_label']}"
        assert output["textblob_label"] is not None, msg

        msg = f"{verbatim} score was expected > 0, but got {output['textblob_score']}"
        assert output["textblob_score"] > 0, msg


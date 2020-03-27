import pytest

import spacy
import en_core_web_sm

from textmap.vectorizers import DocVectorizer
from textmap.tokenizers import SpaCyTokenizer
from .test_common import test_text


def test_spacy_tokenizer():
    tokenizer = SpaCyTokenizer()
    vectorizer = DocVectorizer(tokenizer=tokenizer)
    result = vectorizer.fit(test_text)

def test_spacy_add_sentencizer():
    nlp = en_core_web_sm.load()
    # Remove all of the components
    nlp.remove_pipe("tagger")
    nlp.remove_pipe("parser")
    nlp.remove_pipe("ner")

    tokenizer = SpaCyTokenizer(nlp=nlp)

    assert "sentencizer" in tokenizer.nlp.pipe_names

import pytest

import spacy
from spacy.lang.en import English

from textmap.vectorizers import DocVectorizer
from textmap.tokenizers import SpaCyTokenizer
from .test_common import test_text


def test_spacy_tokenizer():
    tokenizer = SpaCyTokenizer()
    vectorizer = DocVectorizer(tokenizer=tokenizer)
    result = vectorizer.fit(test_text)

def test_spacy_add_sentencizer():
    nlp = English()
    # Remove all of the components
    for p in nlp.pipe_names:
        nlp.remove_pipe(p)

    tokenizer = SpaCyTokenizer(nlp=nlp)

    assert "sentencizer" in tokenizer.nlp.pipe_names

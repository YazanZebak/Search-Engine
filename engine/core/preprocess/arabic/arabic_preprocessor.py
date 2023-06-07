import re
import string
import dateparser

import numpy as np
import pyarabic.trans

from pyarabic import araby
from pyarabic.araby import is_arabicrange
from pyarabic.normalize import normalize_searchtext

from nltk.corpus import stopwords
from nltk.stem.snowball import ArabicStemmer

from engine.core.preprocess.arabic.arabic_lemmatizer import ArabicLemmatizer


class ArabicPreprocessor:
    def __init__(self):
        self.tokenizer = araby
        self.stopwords_tokens = set(stopwords.words('arabic'))
        self.stemmer = ArabicStemmer()
        self.lemmatizer = ArabicLemmatizer()

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans(' ', ' ', string.punctuation))

    def remove_apostrophe(self, text: str) -> str:
        return str(np.char.replace(text, "'", " "))

    def remove_whitespaces(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text)

    def remove_urls(self, text: str) -> str:
        return re.sub("http[^\s]*", "", text, flags=re.IGNORECASE)

    def remove_numbers(self, text: str) -> str:
        return re.sub(r"\d+|[\u0660-\u0669]+", " ", text)

    def remove_stop_words(self, text: str) -> str:
        words = self.tokenizer.tokenize(text, conditions=is_arabicrange, morphs=normalize_searchtext)
        new_text = []
        for w in words:
            if w not in self.stopwords_tokens and len(w) > 1:
                new_text.append(w)
        return ' '.join(new_text)

    def convert_arabic_numbers(self, text):
        return pyarabic.trans.normalize_digits(text, source='all', out='west')

    def stemming(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text, conditions=is_arabicrange, morphs=normalize_searchtext)
        new_text = []
        for w in tokens:
            new_text.append(self.stemmer.stem(w))
        return ' '.join(new_text)

    def lemmatizing(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text, conditions=is_arabicrange, morphs=normalize_searchtext)
        new_text = []
        for token in tokens:
            new_text.append(self.lemmatizer.lemmatize(token))
        return ' '.join(str(word) for word in new_text)

    def preprocess(self, text: str) -> str:
        operations = [
            self.remove_whitespaces,
            self.remove_punctuation,
            self.remove_apostrophe,
            # self.remove_urls,
            self.remove_stop_words,
            self.convert_arabic_numbers,
            self.stemming,
            self.lemmatizing,
        ]

        new_text = text
        for op in operations:
            new_text = op(new_text)

        return new_text

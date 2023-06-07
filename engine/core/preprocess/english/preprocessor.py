import re
import string
import dateparser
import json
import numpy as np
from nltk import tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from engine.core.preprocess.english.lemmatizer import Lemmatizer
from engine.utils.files_paths import SHORTCUTS_PATH


class TextPreprocessor:
    def __init__(self):
        self.tokenizer = tokenize.word_tokenize
        self.stopwords_tokens = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = Lemmatizer()
        with open(SHORTCUTS_PATH) as file:
            self.shortcuts = json.load(file)

    def to_lower(self, text: str) -> str:
        return str(np.char.lower(text))

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
        words = self.tokenizer(text)
        new_text = []
        for w in words:
            if w not in self.stopwords_tokens and len(w) > 1:
                new_text.append(w)
        return ' '.join(new_text)

    def normalize_dates(self, text: str) -> str:
        str_pattern = [
            "\\d{2}-\\d{2}-\\d{4}",
            "[0-9]{2}/{1}[0-9]{2}/{1}[0-9]{4}",
            "\\d{1,2}-(January|February|March|April|May|June|July|August|September|October|November|December)-\\d{4}",
            "\\d{4}-\\d{1,2}-\\d{1,2}",
            "[0-9]{1,2}\\s(January|February|March|April|May|June|July|August|September|October|November|December)\\s\\d{4}",
            "\\d{1,2}-\\d{1,2}-\\d{4}"
        ]

        normalized_text = text
        for pattern in str_pattern:
            for match in re.finditer(pattern, normalized_text):
                parsed_date = dateparser.parse(match.group(), settings={'RETURN_AS_TIMEZONE_AWARE': False})
                normalized_date = parsed_date.strftime('%d %B %Y')
                normalized_text = normalized_text.replace(match.group(), normalized_date)
        return normalized_text

    def replace_shortcuts(self, text):
        pattern = r'\b(' + '|'.join(re.escape(key) for key in self.shortcuts.keys()) + r')\b'
        replaced_text = re.sub(pattern, lambda match: self.shortcuts[match.group(0)], text)
        return replaced_text

    def stemming(self, text: str) -> str:
        tokens = self.tokenizer(text)
        new_text = []
        for w in tokens:
            new_text.append(self.stemmer.stem(w))
        return ' '.join(new_text)

    def lemmatizing(self, text: str) -> str:
        tokens = self.tokenizer(text)
        tagged_tokens = pos_tag(tokens)
        new_text = []
        for token, tag in tagged_tokens:
            new_text.append(self.lemmatizer.lemmatize(token, tag))
        return ' '.join(new_text)

    def preprocess(self, text: str) -> str:

        operations = [
            self.remove_whitespaces,
            self.remove_punctuation,
            self.remove_apostrophe,
            self.to_lower,
            self.remove_stop_words,
            self.normalize_dates,
            self.replace_shortcuts,
            self.stemming,
            self.lemmatizing,
        ]

        new_text = text
        for op in operations:
            new_text = op(new_text)

        return new_text

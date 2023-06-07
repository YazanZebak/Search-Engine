import spacy


class FrenchLemmatizer:
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")

    def lemmatize(self, word: str) -> str:
        doc = self.nlp(word)
        token = doc[0]
        return token.lemma_

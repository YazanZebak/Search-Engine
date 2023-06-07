import qalsadi.lemmatizer


class ArabicLemmatizer:
    def __init__(self):
        self.lemmatizer = qalsadi.lemmatizer.Lemmatizer()

    def lemmatize(self, word: str, pos: str = "") -> str:
        return self.lemmatizer.lemmatize(word)



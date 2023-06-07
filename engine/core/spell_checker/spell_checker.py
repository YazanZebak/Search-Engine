from textblob import TextBlob


class SpellChecker:
    @staticmethod
    def correct(text: str) -> str | None:
        corrected_text = str(TextBlob(text).correct())
        return corrected_text

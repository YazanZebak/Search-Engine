from ar_corrector.corrector import Corrector


class ArabicSpellChecker:
    @staticmethod
    def correct(text: str) -> str | None:
        corr = Corrector()
        corrected_text = corr.contextual_correct(text)
        return corrected_text

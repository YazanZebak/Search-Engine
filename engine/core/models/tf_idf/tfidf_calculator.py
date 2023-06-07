from sklearn.feature_extraction.text import TfidfVectorizer

from engine.utils.file_handler import FileHandler
from engine.utils.files_paths import TF_IDF_MODEL_PATH, TF_IDF_MATRIX_PATH


class TfidfCalculator:
    def __init__(self, documents, dataset_name):
        self.dataset_name = dataset_name
        self.documents = documents
        self.vectorizer = TfidfVectorizer()

    def calculate_tfidf_matrix(self):
        print("Calculating TF-IDF Matrix...")

        tfidf_matrix = self.vectorizer.fit_transform(list(self.documents.values()))

        print("Calculating TF-IDF Matrix Has Been Done Successfully")

        FileHandler.save_model(self.vectorizer, TF_IDF_MODEL_PATH(self.dataset_name))

        FileHandler.save_model(tfidf_matrix, TF_IDF_MATRIX_PATH(self.dataset_name))

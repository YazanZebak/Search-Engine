import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

from engine.utils.file_handler import FileHandler
from engine.utils.files_paths import WORD_2_VEC_MODEL_PATH, WORD_2_VEC_MATRIX_PATH


class Word2VecCalculator:
    def __init__(self, documents, dataset_name):
        self.documents = documents
        self.dataset_name = dataset_name
        self.model = Word2Vec()

    def train_word2vec_model(self):
        print("Training Word2Vec Model...")

        sentences = self.documents.values()
        tokenized_data = [sentence.split() for sentence in sentences]
        self.model.build_vocab(tokenized_data)
        self.model.train(tokenized_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        print("Training Word2Vec Model Has Been Done Successfully")

        FileHandler.save_w2v_model(self.model, WORD_2_VEC_MODEL_PATH(self.dataset_name))

    def calculate_document_vector_matrix(self):
        print("Calculating Word-2-Vec Matrix...")

        doc_ids = list(self.documents.keys())
        document_vectors = np.zeros((len(doc_ids), self.model.vector_size))

        for i, doc_id in enumerate(doc_ids):
            doc_tokens = self.documents[doc_id].split()
            doc_vector = np.zeros(self.model.vector_size)
            count = 0

            for token in doc_tokens:
                if token in self.model.wv:
                    doc_vector += self.model.wv[token]
                    count += 1

            if count > 0:
                doc_vector /= count

            document_vectors[i] = doc_vector

        normalized_vectors = normalize(document_vectors)

        print("Calculating Word-2-Vec Matrix Has Been Done Successfully")

        FileHandler.save_model(normalized_vectors, WORD_2_VEC_MATRIX_PATH(self.dataset_name))

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engine.utils.file_handler import FileHandler
from engine.utils.files_paths import WORD_2_VEC_MODEL_PATH, WORD_2_VEC_MATRIX_PATH, INVERTED_INDEX_PATH


class Word2VecEngine:
    def __init__(self, documents, dataset_name):
        self.inverted_index = FileHandler.read_inverted_index(INVERTED_INDEX_PATH(dataset_name))
        self.word2vec_model = FileHandler.load_w2v_model(WORD_2_VEC_MODEL_PATH(dataset_name))
        self.document_vectors = FileHandler.load_model(WORD_2_VEC_MATRIX_PATH(dataset_name))
        self.documents = documents
        self.doc_index_map = {id: index for index, id in enumerate(documents.keys())}

    def calculate_similarities(self, query):

        query_vector = self.calculate_query_vector(query)
        query_vector = np.array(query_vector).reshape(1, -1)

        similarities = {}

        terms = query.split()

        for term in terms:
            if term in self.inverted_index:

                for doc_index in self.inverted_index[term]:
                    if doc_index not in similarities:
                        doc_id = self.doc_index_map[doc_index]
                        doc_vector = self.document_vectors[doc_id]

                        if np.count_nonzero(doc_vector) == 0:
                            similarity = 0.0
                        else:
                            doc_vector = np.array([doc_vector])
                            similarity = cosine_similarity(query_vector, doc_vector)[0][0]

                        similarities[doc_index] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities

    def calculate_query_vector(self, query):
        vector_sum = np.zeros(self.word2vec_model.vector_size)
        vector_count = 0

        terms = query.split()

        for token in terms:
            if token in self.word2vec_model.wv:
                vector_sum += self.word2vec_model.wv[token]
                vector_count += 1

        if vector_count > 0:
            return vector_sum / vector_count
        return np.zeros(self.word2vec_model.vector_size)

    def retrieve_similar_documents(self, similarities, limit=10):
        similar_documents = []
        count = 0

        for doc_index, similarity in similarities:
            if count >= limit:
                break
            if similarity != 0.0 and self.documents[doc_index] != '':
                similar_documents.append(self.documents[doc_index])
                count += 1

        return similar_documents

    def retrieve_documents_ids(self, similarities):
        documents_ids = []

        for doc_index, similarity in similarities:
            if similarity != 0.0 and self.documents[doc_index] != '':
                documents_ids.append(doc_index)

        return documents_ids

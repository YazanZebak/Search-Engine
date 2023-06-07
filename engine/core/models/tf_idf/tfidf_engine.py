from sklearn.metrics.pairwise import cosine_similarity

from engine.utils.file_handler import FileHandler
from engine.utils.files_paths import TF_IDF_MODEL_PATH, TF_IDF_MATRIX_PATH, INVERTED_INDEX_PATH


class TfidfEngine:
    def __init__(self, documents, dataset_name):
        self.inverted_index = FileHandler.read_inverted_index(INVERTED_INDEX_PATH(dataset_name))
        self.vectorizer = FileHandler.load_model(TF_IDF_MODEL_PATH(dataset_name))
        self.tfidf_matrix = FileHandler.load_model(TF_IDF_MATRIX_PATH(dataset_name))
        self.documents = documents
        self.doc_index_map = {id: index for index, id in enumerate(documents.keys())}

    def calculate_similarities(self, query):

        query_vector = self.calculate_query_vector(query)
        similarities = {}

        terms = query.split()

        for term in terms:
            if term in self.inverted_index:

                for doc_index in self.inverted_index[term]:
                    if doc_index not in similarities:
                        doc_id = self.doc_index_map[doc_index]
                        doc_row = self.tfidf_matrix[doc_id]

                        similarity = cosine_similarity(query_vector, doc_row)[0][0]
                        similarities[doc_index] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_similarities

    def calculate_query_vector(self, query):
        return self.vectorizer.transform([query])

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

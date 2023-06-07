from engine.core.models.tf_idf.tfidf_engine import TfidfEngine
from engine.core.preprocess.english.preprocessor import TextPreprocessor
from engine.evaluation.evaluation_calculator import EvaluationCalculator
from engine.utils.data_factory import DataFactory
from engine.utils.file_handler import FileHandler
from engine.utils.files_paths import WIKIR_DOCUMENTS_PATH, WIKIR_NAME, WIKIR_QUERIES_PATH, WIKIR_QRELS_PATH

if __name__ == '__main__':
    corpus = FileHandler.read_csv_file(WIKIR_DOCUMENTS_PATH)

    queries = FileHandler.read_csv_file(WIKIR_QUERIES_PATH, 10)

    relevance_judgments = FileHandler.read_qrels_file(WIKIR_QRELS_PATH)

    tfidf_engine = TfidfEngine(corpus, WIKIR_NAME)

    processor = TextPreprocessor()

    data_factory = DataFactory(processor)

    retrieved_docs_list = []  # List of retrieved document IDs for each query
    relevant_docs_list = []  # List of relevant document IDs for each query

    print("Testing search engine with queries...")

    for query_id in queries:
        query = queries[query_id]
        query = data_factory.create_processed_text(query)

        similarities = tfidf_engine.calculate_similarities(query)

        retrieved_docs_ids = tfidf_engine.retrieve_documents_ids(similarities)
        retrieved_docs_list.append(retrieved_docs_ids)

        relevant_docs = relevance_judgments.get(int(query_id), {})

        relevant_docs_ids = [doc_id for doc_id in relevant_docs if relevant_docs[doc_id] >= 1]

        relevant_docs_list.append(relevant_docs_ids)

    print("Testing Has Done.")

    print("Calculating Measures...")

    precision_10 = EvaluationCalculator.precision_at_k(retrieved_docs_list[0], relevant_docs_list[0], 10)
    recall = EvaluationCalculator.recall(retrieved_docs_list[0], relevant_docs_list[0])
    map_score = EvaluationCalculator.mean_average_precision(retrieved_docs_list, relevant_docs_list)
    mrr_score = EvaluationCalculator.mean_reciprocal_rank(retrieved_docs_list, relevant_docs_list)

    print("Precision@10:", precision_10)
    print("Recall:", recall)
    print("MAP:", map_score)
    print("MRR:", mrr_score)

class EvaluationCalculator:
    def __init__(self):
        pass

    @staticmethod
    def precision_at_k(retrieved_docs, relevant_docs, k):
        num_relevant = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
        return num_relevant / k

    @staticmethod
    def recall(retrieved_docs, relevant_docs):
        num_relevant = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
        num_total_relevant = len(relevant_docs)
        return num_relevant / num_total_relevant

    @staticmethod
    def average_precision(retrieved_docs, relevant_docs):
        precision_sum = 0.0
        num_relevant = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_sum += EvaluationCalculator.precision_at_k(retrieved_docs, relevant_docs, i + 1)

        if num_relevant == 0:
            return 0.0
        return precision_sum / num_relevant

    @staticmethod
    def mean_average_precision(retrieved_docs_list, relevant_docs_list):
        num_queries = len(retrieved_docs_list)
        avg_precision_sum = 0.0
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            avg_precision_sum += EvaluationCalculator.average_precision(retrieved_docs, relevant_docs)

        if num_queries == 0:
            return 0.0
        return avg_precision_sum / num_queries

    @staticmethod
    def reciprocal_rank(retrieved_docs, relevant_docs):
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs_list, relevant_docs_list):
        num_queries = len(retrieved_docs_list)
        mrr_sum = 0.0
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            mrr_sum += EvaluationCalculator.reciprocal_rank(retrieved_docs, relevant_docs)
        if num_queries == 0.0:
            return 0.0
        return mrr_sum / num_queries

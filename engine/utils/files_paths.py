import os

# Define the base directory path
base_directory = r"C:\Users\ralze_jp3j8n6\PycharmProjects\information-retrieval"

# Dataset Names
TYDI_NAME = 'mr-tydi'
WIKIR_NAME = 'wikir'
FR_WIKIR_NAME = 'fr-wikir'

# Dataset Files
WIKIR_DOCUMENTS_PATH = os.path.join(base_directory, 'datasets', 'wikIR1k', 'documents.csv')
WIKIR_PROCESSED_PATH = os.path.join(base_directory, 'datasets', 'wikIR1k', 'processed_documents.csv')
WIKIR_QUERIES_PATH = os.path.join(base_directory, 'datasets', 'wikIR1k', 'test', 'queries.csv')
WIKIR_QRELS_PATH = os.path.join(base_directory, 'datasets', 'wikIR1k', 'test', 'qrels')

FR_WIKIR_DOCUMENTS_PATH = os.path.join(base_directory, 'datasets', 'FRwikIR14k', 'documents.csv')
FR_WIKIR_PROCESSED_PATH = os.path.join(base_directory, 'datasets', 'FRwikIR14k', 'processed_documents.csv')
FR_WIKIR_QUERIES_PATH = os.path.join(base_directory, 'datasets', 'FRwikIR14k', 'test', 'queries.csv')
FR_WIKIR_QRELS_PATH = os.path.join(base_directory, 'datasets', 'FRwikIR14k', 'test', 'qrels')

TYDI_DOCUMENTS_PATH = os.path.join(base_directory, 'datasets', 'mrtydi-ar', 'collection', 'docs.jsonl')
TYDI_PROCESSED_PATH = os.path.join(base_directory, 'datasets', 'mrtydi-ar', 'collection', 'processed_docs.csv')
TYDI_QUERIES_PATH = os.path.join(base_directory, 'datasets', 'mrtydi-ar', 'topic.dev.tsv')
TYDI_QRELS_PATH = os.path.join(base_directory, 'datasets', 'mrtydi-ar', 'qrels.dev.txt')

SHORTCUTS_PATH = os.path.join(base_directory, 'datasets', 'shortcuts.json')


# Models Files
def INVERTED_INDEX_PATH(dataset_name):
    return os.path.join(base_directory, 'output', dataset_name, 'inverted_index.txt')


def TF_IDF_MATRIX_PATH(dataset_name):
    return os.path.join(base_directory, 'output', dataset_name, 'models', 'tf-idf', 'matrix.pkl')


def TF_IDF_MODEL_PATH(dataset_name):
    return os.path.join(base_directory, 'output', dataset_name, 'models', 'tf-idf', 'model.pkl')


def WORD_2_VEC_MATRIX_PATH(dataset_name):
    return os.path.join(base_directory, 'output', dataset_name, 'models', 'word2vec', 'matrix.pkl')


def WORD_2_VEC_MODEL_PATH(dataset_name):
    return os.path.join(base_directory, 'output', dataset_name, 'models', 'word2vec', 'model.pkl')

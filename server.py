from flask import Flask, jsonify, request
from flask_cors import CORS

from engine.core.models.tf_idf.tfidf_engine import TfidfEngine
from engine.core.models.word2vec.word2vec_engine import Word2VecEngine
from engine.core.preprocess.arabic.arabic_preprocessor import ArabicPreprocessor
from engine.core.preprocess.english.preprocessor import TextPreprocessor
from engine.core.preprocess.french.french_preprocessor import FrenchPreprocessor
from engine.core.spell_checker.arabic_spell_checker import ArabicSpellChecker
from engine.core.spell_checker.spell_checker import SpellChecker

from engine.utils.data_factory import DataFactory
from engine.utils.file_handler import FileHandler
from engine.utils.files_paths import WIKIR_DOCUMENTS_PATH, WIKIR_NAME, TYDI_DOCUMENTS_PATH, TYDI_NAME, \
    FR_WIKIR_DOCUMENTS_PATH, FR_WIKIR_NAME

app = Flask(__name__)
CORS(app)


@app.route('/choose-dataset', methods=["POST"])
def choose_dataset():
    global corpus
    global engine
    global spell_checker
    global dataset_name

    payload = request.get_json()
    dataset = payload.get('dataset')
    dataset_name = dataset

    if dataset == WIKIR_NAME:
        corpus = FileHandler.read_csv_file(WIKIR_DOCUMENTS_PATH)
        engine = Word2VecEngine(corpus, WIKIR_NAME)
        spell_checker = SpellChecker()

    elif dataset == TYDI_NAME:
        corpus = FileHandler.read_jsonl_file(TYDI_DOCUMENTS_PATH, 15000)
        engine = Word2VecEngine(corpus, TYDI_NAME)
        spell_checker = ArabicSpellChecker()

    else:
        corpus = FileHandler.read_csv_file(FR_WIKIR_DOCUMENTS_PATH, 15000)
        engine = Word2VecEngine(corpus, FR_WIKIR_NAME)
        spell_checker = SpellChecker()

    response = {
        "status": True,
        "data": "Dataset (" + dataset + ") Has Been Uploaded Successfully."
    }

    return jsonify(response)


@app.route('/correct', methods=["POST"])
def correct():
    payload = request.get_json()
    query = payload.get('query')

    query = spell_checker.correct(query)

    response = {
        "status": True,
        "query": query
    }

    return jsonify(response)


@app.route('/search', methods=["POST"])
def index():
    payload = request.get_json()
    query = payload.get('query')

    if dataset_name == WIKIR_NAME:
        processor = TextPreprocessor()
    elif dataset_name == TYDI_NAME:
        processor = ArabicPreprocessor()
    else:
        processor = FrenchPreprocessor()

    data_factory = DataFactory(processor)

    query = data_factory.create_processed_text(query)

    similarities = engine.calculate_similarities(query)

    results = engine.retrieve_similar_documents(similarities)

    response = {
        "status": True,
        "documents": results
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=False)

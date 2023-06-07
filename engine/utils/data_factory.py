from collections import defaultdict
from engine.utils.file_handler import FileHandler


class DataFactory:
    def __init__(self, preprocessor):
        self._processor = preprocessor

    def create_processed_text(self, text):
        return self._processor.preprocess(text)

    def create_processed_data(self, corpus):
        print("Preprocessing Documents...")
        documents = defaultdict(dict)

        for id, txt in corpus.items():
            print(id)
            documents[id] = self._processor.preprocess(txt)

        print("Preprocessing Has Been Done Successfully")
        return documents

    def create_inverted_index(self, documents, index_path):
        print("Creating Inverted Index...")
        inverted_index = defaultdict(list)

        for key, value in documents.items():
            txt = value
            tokens = txt.split()
            for token in tokens:
                if key not in inverted_index[token]:
                    inverted_index[token].append(key)

        print("Creating Inverted Index Has Been Done Successfully")

        FileHandler.write_inverted_index(inverted_index, index_path)

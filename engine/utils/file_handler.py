import csv
import json
import pickle
from collections import defaultdict
from gensim.models import Word2Vec

MAX_LIMIT = 150000


class FileHandler:
    def __init__(self):
        pass

    @staticmethod
    def read_csv_file(file_path, limit=MAX_LIMIT, delimiter=','):
        documents = defaultdict(dict)
        print(f"Reading data from {file_path} ...")
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            next(reader)  # skip the header row
            row_count = 0
            for row in reader:
                id = row[0]
                txt = row[1]
                documents[id] = txt
                row_count += 1
                if row_count >= limit:
                    break

        print(f"Read {len(documents)} documents")
        return documents

    @staticmethod
    def write_csv_file(file_path, documents):
        print(f"Writing data to {file_path} ...")
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Text"])  # write the header row
            for id, txt in documents.items():
                writer.writerow([id, txt])

        print("Data has been written successfully")

    @staticmethod
    def read_qrels_file(file_path, delimiter='\t'):
        print(f"Reading data from {file_path} ...")
        relevance_judgments = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                query_id, _, doc_id, relevance = line.split(delimiter)
                query_id = int(query_id)
                doc_id = doc_id
                relevance = int(relevance)

                if query_id not in relevance_judgments:
                    relevance_judgments[query_id] = {}

                relevance_judgments[query_id][doc_id] = relevance
        print(f"Read {len(relevance_judgments)} documents")
        return relevance_judgments

    @staticmethod
    def read_jsonl_file(file_path, limit=MAX_LIMIT):
        documents = defaultdict(dict)
        print(f"Reading data from {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as file:
            row_count = 0
            for line in file:
                data = json.loads(line)
                id = data["id"]
                txt = data["contents"]
                documents[id] = txt
                row_count += 1
                if row_count >= limit:
                    break

        print(f"Read {len(documents)} documents")
        return documents

    @staticmethod
    def read_inverted_index(file_path):
        print(f"Reading inverted index from {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as file:
            inverted_index = {}
            for line in file:
                line = line.strip()
                if line:
                    term, postings_str = line.split(':')
                    postings = [post.strip() for post in postings_str.split(',')]
                    inverted_index[term] = postings
        print("Inverted index has been loaded successfully.")
        return inverted_index

    @staticmethod
    def write_inverted_index(inverted_index, file_path):
        print(f"Writing inverted index to {file_path} ...")
        with open(file_path, 'w', encoding='utf-8') as file:
            for term, postings in inverted_index.items():
                line = f"{term}: {','.join(postings)}\n"
                file.write(line)
        print("Inverted index written to file successfully.")

    @staticmethod
    def save_model(model, file_path):
        print(f"Writing The Model to {file_path} ...")
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print("The Model has been saved successfully.")

    @staticmethod
    def load_model(file_path):
        print(f"Reading The Model from {file_path} ...")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print("The Model has been loaded successfully.")
        return model

    @staticmethod
    def save_w2v_model(model, file_path):
        print(f"Writing The Model to {file_path} ...")
        model.save(file_path)
        print("The Model has been saved successfully.")

    @staticmethod
    def load_w2v_model(file_path):
        print(f"Reading The Model from {file_path} ...")
        model = Word2Vec.load(file_path)
        print("The Model has been loaded successfully.")
        return model

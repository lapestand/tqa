from src.pre_processing.PreProcessor import PreProcessor
from src.Helper import Properties
import numpy as np
import pandas as pd

from multiprocessing import Process

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


class SyntacticAnalyzer:
    def __init__(self, model, sentences):
        self.model = model
        self.document_term_matrix = self.model.document_term_matrix_tfidf()
        self.matrix_columns = self.document_term_matrix.columns.to_numpy().tolist()

        self.real_doc_sentences = sentences

        self.q_tokens = list()

    def answer(self, question, is_parsed=False):
        if not is_parsed:
            self.parse_question(question)
        else:
            self.q_tokens = question
        # print(self.q_tokens)
        # print(self.document_term_matrix)
        if Properties.distance_mode["euclidian_distance"]:  # NOT WORKING
            ed = self.get_euclidian_distances()
            index_min = min(range(len(ed)), key=ed.__getitem__)
            return self.real_doc_sentences[index_min], index_min
        else:  # COSINE SIMILARITIES
            cs = self.get_cosine_similarities()
            index_max = max(range(len(cs)), key=cs.__getitem__)
            return self.real_doc_sentences[index_max], index_max

    # def __parse_question(self, question):
    #     p = PreProcessor(question, "empty", "empty", is_question=True)
    #     p.pre_process()
    #     self.preprocessor_for_question = p
    #     self.q_tokens.append(p.lemmas["tuple"][0])

    def parse_question(self, question):
        p = PreProcessor(is_question=True)
        p.pre_process(question, "txt")
        self.q_tokens = p.data["p_data"]["pos"]["tuple"]

    def get_euclidian_distances(self):
        query_tfidf = self.model.vectorizer.transform(self.q_tokens)
        ed = euclidean_distances(query_tfidf, self.model.vectors).flatten()
        print(f"Euclidian distances: {ed}")
        return ed

    def convert2vector(self, tokens):
        t = dict.fromkeys(self.matrix_columns)
        for token in tokens:
            if token in t:
                t[token] += 1
        return pd.DataFrame([t.values()], columns=self.matrix_columns)

    def get_cosine_similarities(self):
        query_tfidf = self.model.vectorizer.transform(self.q_tokens)
        # df = pd.DataFrame(query_tfidf.todense(), index=range(1),
        #                   columns=self.model.vectorizer.get_feature_names())
        print(f"Cosine similarity: {cosine_similarity(query_tfidf, self.model.vectors).flatten()}")
        return cosine_similarity(query_tfidf, self.model.vectors).flatten()

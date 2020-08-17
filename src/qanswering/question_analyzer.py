from src.pre_processing.PreProcessor import PreProcessor
from src.Helper import Properties
import numpy as np
import pandas as pd

from multiprocessing import Process

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Analyser:
    def __init__(self, model, sentences):
        # print(f"\n\nProcessing data: \n\t{p.processing_data}"
        #       f"\n\nParagraphs: \n\t{p.paragraphs}"
        #       f"\n\nSentences: \n\t{p.sentences}"
        #       f"\n\nFiltered Sentences: \n\t{p.filtered_sentences}"
        #       f"\n\nTokens: \n\t{p.tokens}"
        #       f"\n\nLemmas: \n\t{p.lemmas_of_sentences}"
        #       )
        # print(f"Tokens: {p.lemmas_of_sentences}")
        self.model = model
        self.document_term_matrix = self.model.document_term_matrix_tfidf()
        self.matrix_columns = self.document_term_matrix.columns.to_numpy().tolist()

        self.real_doc_sentences = sentences

        self.preprocessor_for_question = PreProcessor
        self.tokens = list()
        self.q_tokens = list()

    def answer(self, question, q_order):
        if type(question) == str:
            question = [question]
        # self.__parse_question(questions)

        if Properties.analyzer_mode["euclidian_distance"]:
            return min(self.get_euclidian_distances(self.convert2vector(self.q_tokens[q_order])))
        else:
            cosine_similarities = self.get_cosine_similarities(q_order)
            # print(cosine_similarities)
            index_max = max(range(len(cosine_similarities)), key=cosine_similarities.__getitem__)
            return self.real_doc_sentences[index_max]

    # def __parse_question(self, question):
    #     p = PreProcessor(question, "empty", "empty", is_question=True)
    #     p.pre_process()
    #     self.preprocessor_for_question = p
    #     self.q_tokens.append(p.lemmas["tuple"][0])

    def parse_questions(self, questions):
        if type(questions) == str:
            questions = [questions]
        p = PreProcessor(" ".join(questions), "empty", "empty", is_question=True)
        p.pre_process()
        self.preprocessor_for_question = p
        self.q_tokens = p.lemmas["tuple"]

    def get_euclidian_distances(self, relevant_tokens):
        return [self.euclidian_distance(relevant_tokens, self.document_term_matrix[idx]) for idx in
                range(len(self.document_term_matrix))]

    def euclidian_distance(self, v1, v2):
        pass

    def convert2vector(self, tokens):
        t = dict.fromkeys(self.matrix_columns)
        for token in tokens:
            if token in t:
                t[token] += 1
        return pd.DataFrame([t.values()], columns=self.matrix_columns)

    def get_cosine_similarities(self, q_order):
        query_tfidf = self.model.vectorizer.transform([self.q_tokens[q_order]])
        # df = pd.DataFrame(query_tfidf.todense(), index=range(1),
        #                   columns=self.model.vectorizer.get_feature_names())
        return cosine_similarity(query_tfidf, self.model.vectors).flatten()

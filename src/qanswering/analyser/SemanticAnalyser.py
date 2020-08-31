from src.Helper import Properties
from src.qanswering.analyser.SyntacticAnalyser import SyntacticAnalyzer
from src.modelling.Model import Model

import pandas as pd
import numpy as np


class RuleBasedHybridAnalyzer:
    def __init__(self, model, sentences, pos):
        self.model = model
        self.real_doc_sentences = sentences
        # self.adjectives = Properties.question_adjectives
        self.sentence_pos = pos

    def answer(self, question, q_order):
        syntactic_analyzer = SyntacticAnalyzer(self.model, self.real_doc_sentences)
        syntactic_analyzer.parse_question(question)
        q_tokens = syntactic_analyzer.q_tokens
        relevant_docs_idx = self.find_relevant_answers(self.model, q_tokens)
        relevant_docs = list()
        relevant_pos = list()
        for idx in relevant_docs_idx:
            relevant_docs.append(self.real_doc_sentences[idx])
            relevant_pos.append(self.sentence_pos[idx])
        # relevant_docs_doc = " ".join(relevant_docs)
        # print(relevant_docs_doc)
        # print(type(self.sentence_pos))
        # print(self.sentence_pos)
        # irrelevant_idx = [idx for idx in range(self.model.document_term_matrix_tfidf().shape[0]) if
        #                   idx not in relevant_answers_idx]

        relevant_model = Model(relevant_pos)
        syntactic_analyzer = SyntacticAnalyzer(relevant_model, relevant_docs)
        return syntactic_analyzer.answer(q_tokens, is_parsed=True)

    def find_relevant_answers(self, model, q_tokens):
        # a = [('kaç', 'Adj'), ('kaçta', 'Adv'),
        #      ('kaçıncı', 'Adj'), ('nasıl', 'Adj'), ('nasıl', 'Adv'),
        #      ('ne', 'Adj'), ('ne', 'Pron'), ('neredeki', 'Pron')]

        """
        If any element of Properties.questionPOS["Num"] in q_tokens find indices of relevant sentences
        Note: bigger loop inside {speed optimization}  
        """
        rows, cols = model.document_term_matrix_tfidf().shape
        relevant_rows = list(range(rows))
        if any(adj in max(Properties.questionPOS["Num"], q_tokens[0], key=len) for adj in
               min(Properties.questionPOS["Num"], q_tokens[0], key=len)):
            relevant_rows = self.find_rows(model, looking_for="Num")
        return relevant_rows
        # for row in relevant_rows:
        #     print(self.real_doc_sentences[row])
        # exit(1)

    def find_rows(self, model, looking_for="Num"):
        """
        :param model: Model is document term matrix which calculated using tf(Augmented)-idf(Smoothed)
        :param looking_for: Pos value which is looking for
        :return: Indices of sentences which include looking_for(default = "Num") value
        """
        columns = model.document_term_matrix_tfidf().columns.to_numpy().tolist()  # Get column names from model matrix
        num_cols = [col for col in columns if looking_for in col]  # Find columns which include looking_for
        df = model.document_term_matrix_tfidf()[num_cols]  # Narrow model matris using num_cols
        return [idx for idx, row in df.iterrows() if np.any(row)]  # Find index of rows which not all values are zero in


class AnnAnalyzer:
    def __init__(self, model, sentences):
        pass

    def answer(self, questions, q_order):
        pass

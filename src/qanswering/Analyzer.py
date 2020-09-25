from src.pre_processing.PreProcessor import PreProcessor
from src.Helper import Properties
import numpy as np
import pandas as pd

from src.qanswering.analyser import SyntacticAnalyzer, SemanticAnalyzer

from multiprocessing import Process

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Analyser:
    def __init__(self, model, sentences, sentence_pos):
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
        self.sentence_pos = sentence_pos
        self.real_doc_sentences = sentences

        self.q_tokens = list()

        if Properties.analyzer_mode["Syntactic"]:
            self.analyzer = SyntacticAnalyzer.SyntacticAnalyzer(self.model, self.real_doc_sentences)
        elif Properties.analyzer_mode["RuleBased_hybrid"]:
            self.analyzer = SemanticAnalyzer.RuleBasedHybridAnalyzer(self.model, self.real_doc_sentences,
                                                                     self.sentence_pos)
        elif Properties.analyzer_mode["ANN"]:
            self.analyzer = SemanticAnalyzer.AnnAnalyzer(self.model, self.real_doc_sentences)
        else:
            raise KeyError(
                "No answer method selected!"
            )

    def answer(self, question):
        return self.analyzer.answer(question)

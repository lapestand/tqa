import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def dummy(text):
    # print(text)
    return text


class Model:
    def __init__(self, docs):
        self.vectorizer = TfidfVectorizer(use_idf=True, lowercase=False, preprocessor=dummy, tokenizer=dummy)
        self.vectors = self.vectorizer.fit_transform(docs)
        self.doc_count = len(docs)

    def __del__(self):
        pass

    def document_term_matrix_tfidf(self):
        return pd.DataFrame(
            self.vectors.todense(), index=range(self.doc_count), columns=self.vectorizer.get_feature_names()
        )

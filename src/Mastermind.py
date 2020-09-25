from src.Helper import Properties
from src.modes import tfidf_pos


class Mastermind:

    def __init__(self, mode):
        self.mode = mode

    def read(self, raw_data, file_type):
        if self.mode == "TfIdf+PoS":
            return tfidf_pos.read(raw_data, file_type)

    def find_answer(self, to, looking):
        answer = str()
        if self.mode == "TfIdf+PoS":
            from src.modes import tfidf_pos
            answer = tfidf_pos.answer(to, looking)[0]

        return answer

import re
import os
import time
import numpy as np
import nltk
from nltk.corpus import stopwords
import multiprocessing
from bs4 import Comment
from os.path import join
from bs4 import BeautifulSoup
from multiprocessing import Process, Value
from ctypes import c_bool
from sklearn.feature_extraction.text import TfidfVectorizer
from jpype import JClass, getDefaultJVMPath, shutdownJVM, startJVM, JString, java
import sys

from src import Helper
from src.Helper import Properties
from src import Helper


class PreProcessor:
    def __init__(self, is_question=False, is_semantic=False):
        """
        RETURN LIST:
            ""
            full_document
            raw_sentences
            filtered_sentences
            tokens
            pos = {
                    "mini": {
                        "tuple":list(),
                        "str":list(),
                        "sentence":list()
                    },
                    "full": {
                        "tuple":list(),
                        "str":list(),
                        "sentence":list()
                    }
                }
            ""
        :param raw_data:
        :param is_question:
        :param is_semantic:
        """
        self.is_question = is_question
        self.is_semantic = is_semantic

        self.data = {
            "raw_data": {
                "doc": None,
                "paragraphs": list(),
                "sentences": list()
            },
            "p_data": {
                "decoded_doc": None,
                "lowercase_sentences": list(),
                "filtered_sentences": list(),
                "tokens": list(),
                "pos": {
                    # "mini": {"raw": list(), "tuple": list(), "str": list(), "sentence": list()},
                    # "full": {"raw": list(), "tuple": list(), "str": list(), "sentence": list()}
                    "raw": list(),
                    "tuple": list(),
                    "str": list(),
                    "sentence": list()
                }
            }
        }

    def pre_process(self, rd, fd, remove_stop_words=False):
        Helper.debug("Pre Processor", 1, "situation")
        """
        :param rd: raw data
        :param fd: file type
        :return: self.data
        """
        # DECODING
        if not self.is_question:
            self.data["raw_data"]["doc"] = rd
            self.data["p_data"]["decoded_doc"] = self.decode(rd, str(fd))
        else:
            self.data["p_data"]["decoded_doc"] = rd
        ##########

        if self.split_into_paragraphs(str(self.data["p_data"]["decoded_doc"])):  # SPLIT INTO PARAGRAPHS
            Helper.debug("split into paragraphs", True, "module_debug")
        else:
            Helper.debug("split into paragraphs", False, "module_debug")

        if self.split_into_sentences(self.data["raw_data"]["paragraphs"]):  # SPLIT INTO SENTENCES
            Helper.debug("split into sentences", True, "module_debug")
        else:
            Helper.debug("split into sentences", False, "module_debug")

        self.data["p_data"]["lowercase_sentences"] = [_.lower() for _ in
                                                      (self.data["raw_data"]["sentences"])]  # CONVERT LOWER CASE
        Helper.debug("lower_case", True, "module_debug")

        if self.word_tokenization(self.data["p_data"]["lowercase_sentences"]):  # TOKENIZATION
            Helper.debug("word tokenize", True, "module_debug")
        else:
            Helper.debug("word tokenize", False, "module_debug")

        self.remove_punctuation(self.data["p_data"]["tokens"],
                                self.data["p_data"]["lowercase_sentences"])  # REMOVE PUNCTUATION
        Helper.debug("remove punctuation", True, "module_debug")

        if remove_stop_words:
            if self.remove_stop_words(self.data["p_data"]["lowercase_sentences"]):  # REMOVE STOP WORDS
                Helper.debug("remove stop words", True, "module_debug")
            else:
                Helper.debug("remove stop words", False, "module_debug")
        else:
            Helper.debug("remove stop words", False, "module_debug")

        if self.lemmatization(self.data["p_data"]["filtered_sentences"]):
            Helper.debug("lemmatization", True, "module_debug")
        else:
            Helper.debug("lemmatization", False, "module_debug")

    def lemmatization(self, sentences):
        def run(ss, rt_lemmas, rt):
            try:
                # sys.stdout = open(os.devnull)
                startJVM(
                    getDefaultJVMPath(),
                    '-ea',
                    f'-Djava.class.path={Properties.zemberek_path}',
                    convertStrings=False
                )

                print("*" * 50, end="\n")
                TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
                Paths: JClass = JClass('java.nio.file.Paths')

                s = time.time()
                morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()
                print(f"\nInitializing time {time.time() - s}")
                print("*" * 50, end="\n")

                for idx, sentence in enumerate(ss):
                    analysis: java.util.ArrayList = morphology.analyzeSentence(sentence)
                    results: java.util.ArrayList = (
                        morphology.disambiguate(sentence, analysis).bestAnalysis()
                    )

                    lemmas_ = list()
                    for i, result in enumerate(results, start=1):
                        if str(min(result.getLemmas(), key=len)) == "UNK":
                            lemmas_.append([sentence.split()[i - 1], str(result.formatLong())])
                        else:
                            lemmas_.append([str(max(result.getLemmas(), key=len)), str(result.formatLong())])
                    if rt_lemmas[idx] is None:
                        rt_lemmas[idx] = lemmas_
                    else:
                        rt.value = False
                shutdownJVM()
            except:
                rt.value = False

        def extract_token_and_pos(lemmas_):
            cell_list = list()
            str_list = list()
            doc = str()
            for lemma in lemmas_:
                token: str = lemma[0]
                pos: str = lemma[1].split("[", 1)[1].split("]")[0].split(":")[1].split(",")[0]

                cell_list.append((token, pos))
                str_list.append(str(token + "|||" + pos))
                doc += token + "|||" + pos + " "
            return cell_list, str_list, doc[:-1]

        try:
            manager = multiprocessing.Manager()

            lemmas = manager.list()
            lemmas.extend([None] * len(sentences))

            ret = Value(c_bool, True)

            p = Process(target=run, args=(self.data["p_data"]["filtered_sentences"], lemmas, ret))
            p.start()
            p.join()

            self.data["p_data"]["pos"]["raw"] = list(lemmas)
            for lemma_sentence in lemmas:
                t_l, s_l, s = extract_token_and_pos(lemma_sentence)
                self.data["p_data"]["pos"]["tuple"].append(t_l)
                self.data["p_data"]["pos"]["str"].append(s_l)
                self.data["p_data"]["pos"]["sentence"].append(s)

            return ret.value
        except:
            return False

    def decode(self, doc, file_type: str) -> str:
        if file_type == "html":
            def tag_visible(element):
                if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                    return False
                if isinstance(element, Comment):
                    return False
                return True

            def text_from_html(body):
                soup = BeautifulSoup(body, 'html.parser')
                texts = soup.findAll(text=True)
                visible_texts = filter(tag_visible, texts)
                return " ".join(t.strip() for t in visible_texts)

            raw_data = doc.decode("utf-8")
            decoded = {
                "info": raw_data[:raw_data.find("<head")],
                "head": raw_data[raw_data.find("<head"):raw_data.find("</head>") + 7],
                # "body": raw_data[raw_data.find("<body"):raw_data.find("</body>") + 7],
                "body": text_from_html(raw_data[raw_data.find("<body"):raw_data.find("</body>") + 7]),
                "footer": raw_data[raw_data.find("<footer"):raw_data.find("</footer>") + 9]
            }["body"]
        elif file_type == "txt":
            decoded = doc
        elif file_type == "pdf":
            decoded = doc
        else:
            raise Exception(f"File type should be in following types {Properties.supported_file_types}."
                            f" The type was: {file_type}")
        return decoded
        # return " ".join(decoded.split())

    def split_into_paragraphs(self, doc: str):
        try:
            self.data["raw_data"]["paragraphs"] = list(filter(lambda x: x != '', doc.split('\n\n')))
            return True
        except:
            return False

    def split_into_sentences(self, paragraphs):
        def run(p, ss, r):
            try:
                startJVM(
                    getDefaultJVMPath(),
                    '-ea',
                    f'-Djava.class.path={Properties.zemberek_path}',
                    convertStrings=False
                )
                TurkishSentenceExtractor: JClass = JClass(
                    'zemberek.tokenization.TurkishSentenceExtractor'
                )

                extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT

                for _ in p:
                    for s in extractor.fromParagraph(_):
                        ss.append(str(s))
                shutdownJVM()
                r.value = True
            except:
                r.value = False

        manager = multiprocessing.Manager()
        sentences = manager.list()
        ret = Value(c_bool, False)
        p_s = Process(target=run, args=(paragraphs, sentences, ret))
        p_s.start()
        p_s.join()
        self.data["raw_data"]["sentences"] = sentences
        return ret

    def word_tokenization(self, sentences):
        def run(sentence, i, rt_tokens, rt):
            try:
                startJVM(
                    getDefaultJVMPath(),
                    '-ea',
                    f'-Djava.class.path={Properties.zemberek_path}',
                    convertStrings=False
                )

                new_tokens = list()
                turkish_tokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
                java_token: JClass = JClass('zemberek.tokenization.Token')

                tokenizer: turkish_tokenizer = turkish_tokenizer.DEFAULT

                inp = sentence

                token_iterator = tokenizer.getTokenIterator(JString(inp))
                for token in token_iterator:
                    new_tokens.append(
                        {
                            "token": str(token),
                            "content": str(token.content),
                            "normalized": str(token.normalized),
                            "type": str(token.type),
                            "start": str(token.start),
                            "end": str(token.end)
                        }
                    )
                if rt_tokens[i] is None:
                    rt_tokens[i] = new_tokens
                else:
                    rt[i] = False
                shutdownJVM()
            except:
                rt[i] = False

        try:
            processes = list()
            manager = multiprocessing.Manager()
            tokens = manager.list()
            ret_values = manager.list()

            ret_values.extend([True] * len(sentences))
            tokens.extend([None] * len(sentences))

            for idx, s1 in enumerate(sentences):
                processes.append(Process(target=run, args=(s1, idx, tokens, ret_values)))

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            self.data["p_data"]["tokens"] = tokens
            return False not in ret_values
        except:
            return False

    def remove_punctuation(self, tokens, sentences):
        def run(sent, tok):
            for p in tok[::-1]:
                start = int(p["start"])
                stop = int(p["end"])
                sent = sent[0: start:] + sent[stop + 1::]
            return sent

        try:
            new_list = list()
            for idx, (token_list, sentence) in enumerate(
                    zip(tokens, sentences)):
                new_list.append(
                    run(sentence, [t for t in token_list if t["type"] == "Punctuation"])
                )
            self.data["p_data"]["filtered_sentences"] = new_list
            return True
        except:
            return False

    def remove_stop_words(self, sentences, stop_words=None):
        if stop_words is None:
            stop_words = set(stopwords.words('turkish'))

        def run(sentence):
            return re.sub(r' (?=\W)', '', ' '.join(
                [w for w in sentence.split() if w not in stop_words]))

        try:
            new_sentences = list()
            for s in sentences:
                new_sentences.append(run(s))
            self.data["p_data"]["filtered"] = new_sentences
            return True
        except:
            return False

import re
import os
import time
import numpy as np
from src import Helper
import nltk
from nltk.corpus import stopwords
import multiprocessing
from bs4 import Comment
from os.path import join
from bs4 import BeautifulSoup
from src.Helper import Properties
from multiprocessing import Process
from sklearn.feature_extraction.text import TfidfVectorizer
from jpype import JClass, getDefaultJVMPath, shutdownJVM, startJVM, JString, java


def sentence_boundary_detection(paragraphs, sentences):
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

        for p in paragraphs:
            for s in extractor.fromParagraph(p):
                sentences.append(str(s))
        shutdownJVM()
        Helper.debug("sentence_boundary_detection", True, "module_debug")
        return True
    except:
        Helper.debug("sentence_boundary_detection", False, "module_debug")
        return False


def word_tokenization(sentence, idx, rt_tokens):
    try:
        startJVM(
            getDefaultJVMPath(),
            '-ea',
            f'-Djava.class.path={Properties.zemberek_path}',
            convertStrings=False
        )

        tokens = list()
        TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
        Token: JClass = JClass('zemberek.tokenization.Token')

        tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT

        inp = sentence

        token_iterator = tokenizer.getTokenIterator(JString(inp))
        for token in token_iterator:
            tokens.append(
                {
                    "token": str(token),
                    "content": str(token.content),
                    "normalized": str(token.normalized),
                    "type": str(token.type),
                    "start": str(token.start),
                    "end": str(token.end)
                }
            )
        if rt_tokens[idx] is None:
            rt_tokens[idx] = tokens
        else:
            print("\n\n\n\n\nAMAZING ERROR\n\n\n\n\n")

        # print("START FROM HERE")
        #
        # print('Default Tokenization Example:\n')
        #
        # tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT
        #
        # print(f'Input = {inp}')
        # for i, token in enumerate(tokenizer.tokenizeToStrings(
        #         JString(inp)
        # )):
        #     print(f' | Token String {i} = {token}')
        #
        # print('\nCustom Tokenization Example:\n')
        #
        # tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
        #     Token.Type.Punctuation,
        #     Token.Type.NewLine,
        #     Token.Type.SpaceTab
        # ).build()
        #
        # inp: str = 'Saat, 12:00'
        # print(f'Input = {inp}')
        # for i, token in enumerate(tokenizer.tokenize(JString(inp))):
        #     print(f' | Token {i} = {token}')
        shutdownJVM()
    except:
        Helper.debug("word_tokenization", False, "module_debug")


def find_lemmas(sentences, rt_lemmas):
    try:
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
        print("*" * 50, end="\n")

        print(f"Initializing time {time.time() - s}")
        for idx, sentence in enumerate(sentences):
            analysis: java.util.ArrayList = morphology.analyzeSentence(sentence)
            results: java.util.ArrayList = (
                morphology.disambiguate(sentence, analysis).bestAnalysis()
            )

            lemmas = list()
            for i, result in enumerate(results, start=1):
                # if (
                #         str(min(result.getLemmas(), key=len)) == "UNK"
                #         or token_list[i]["type"] not in ["Word", "WordAlphanumerical", "WordWithSymbol"]
                # ):
                if str(min(result.getLemmas(), key=len)) == "UNK":
                    lemmas.append([sentence.split()[i - 1], str(result.formatLong())])
                else:
                    # print(str(max(result.getLemmas(), key=len)))
                    lemmas.append([str(max(result.getLemmas(), key=len)), str(result.formatLong())])
                    # print(result.formatLong())
                # print(lemmas[-1])
                # print(
                #     f'\nAnalysis {i}: {str(result.formatLong())}'
                #     f'\nStems {i}:'
                #     f'{", ".join([str(stem) for stem in result.getStems()])}'
                #     f'\nLemmas {i}:'
                #     f'{", ".join([str(stem) for stem in result.getLemmas()])}'
                # )
            if rt_lemmas[idx] is None:
                rt_lemmas[idx] = lemmas
            else:
                print("Collision while lemmatization")
        Helper.debug("lemmatization", True, "module_debug")
        shutdownJVM()
    except:
        Helper.debug("lemmatization", True, "module_debug")
        return False


class PreProcessor:
    def __init__(self, raw_data, f_name, f_type, is_question=False):
        self.is_question=is_question
        if not is_question:
            Helper.debug("Pre Processor", 0, "situation")
            self.decoders = {
                "html": self.__html_decode,
                "txt": self.__txt_decode,
                "pdf": self.__pdf_decode
            }

            self.data = self.decoders[f_type](raw_data)
            self.processing_data = self.data if type(self.data) is str else self.data["body"]

            self.processing_data = " ".join(self.processing_data.split())
        else:
            self.processing_data = raw_data

        self.paragraphs = list()
        self.sentences = list()

        ###
        self.split_into_paragraphs()
        self.__split_into_sentences()
        self.untouched_sentences = self.sentences
        self.paragraphs = list()
        self.sentences = list()
        ###

        self.unpunc_sentences = list()
        self.filtered_sentences = list()
        self.tokens = list()
        self.lemmas = {"raw": list(), "tuple": list(), "str": list(), "sentence": list()}

    def __del__(self):
        pass

    def pre_process(self):
        Helper.debug("Pre Processor", 1, "situation")

        self.processing_data = self.processing_data.lower()  # CONVERT LOWER CASE
        Helper.debug("lower_case", True, "module_debug")

        self.split_into_paragraphs()  # SPLIT INTO PARAGRAPHS
        Helper.debug("split into paragraphs", True, "module_debug")

        self.__split_into_sentences()  # SPLIT INTO SENTENCES
        Helper.debug("split into sentences", True, "module_debug")

        self.word_tokenization()  # PRE-TOKENIZE
        Helper.debug("pre tokenize", True, "module_debug")

        self.remove_punctuation()  # REMOVE PUNCTUATION
        Helper.debug("remove punctuation", True, "module_debug")

        self.remove_stop_words()  # REMOVE STOP WORDS
        Helper.debug("remove stop words", True, "module_debug")

        # processes.append(Process(target=find_lemmas, args=(self.filtered_sentences[-1], idx, lemmas, token_list)))
        # for token in token_list:
        #     print((
        #         f'\tToken = {token["token"]}'
        #         f'\n\t | Content = {token["content"]}'
        #         f'\n\t | Normalized = {token["normalized"]}'
        #         f'\n\t | Type = {token["type"]}'
        #         f'\n\t | Start = {token["start"]}'
        #         f'\n\t | End = {token["end"]}\n'
        #     ))

        # print("Sentence Disambiguation started!")

        self.lemmatization()

        Helper.debug("Pre Processing", 2, "situation")

    def __html_decode(self, raw_data):
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

        raw_data = raw_data.decode("utf-8")
        # print(text_from_html(raw_data[raw_data.find("<body"):raw_data.find("</body>") + 7]))
        return {
            "info": raw_data[:raw_data.find("<head")],
            "head": raw_data[raw_data.find("<head"):raw_data.find("</head>") + 7],
            # "body": raw_data[raw_data.find("<body"):raw_data.find("</body>") + 7],
            "body": text_from_html(raw_data[raw_data.find("<body"):raw_data.find("</body>") + 7]),
            "footer": raw_data[raw_data.find("<footer"):raw_data.find("</footer>") + 9]
        }

    def __txt_decode(self, raw_data):
        return raw_data

    def __pdf_decode(self, raw_data):
        return raw_data

    def __split_into_sentences(self):
        manager = multiprocessing.Manager()
        sentences = manager.list()
        p_s = Process(target=sentence_boundary_detection, args=(self.paragraphs, sentences))
        # processes.append(Process(target=word_tokenization, args=(self.processing_data, tokens)))
        p_s.start()
        p_s.join()
        self.sentences = sentences

    def split_into_paragraphs(self):
        try:
            self.paragraphs = list(filter(lambda x: x != '', self.processing_data.split('\n\n')))
            Helper.debug("split_into_paragraphs", True, "module_debug")
            return True
        except:
            Helper.debug("split_into_paragraphs", False, "module_debug")
            return False

    def remove_stop_words(self):
        def run(sentence):
            return re.sub(r' (?=\W)', '', ' '.join(
                [w for w in sentence.split() if w not in stop_words]))

        stop_words = set(stopwords.words('turkish'))
        for s in self.unpunc_sentences:
            self.filtered_sentences.append(run(s))

    def remove_punc(self, sentence, p_tokens):
        for p in p_tokens[::-1]:
            start = int(p["start"])
            stop = int(p["end"])
            sentence = sentence[0: start:] + sentence[stop + 1::]
        return sentence

    def extract_token_and_pos(self, lemmas):
        cell_list = list()
        str_list = list()
        doc = str()
        for lemma in lemmas:
            token: str = lemma[0]
            pos: str = lemma[1].split("[", 1)[1].split("]")[0].split(":")[1].split(",")[0]

            cell_list.append((token, pos))
            str_list.append(str(token + "|||" + pos))
            doc += token + "|||" + pos + " "
        return cell_list, str_list, doc[:-1]

    def word_tokenization(self):
        manager = multiprocessing.Manager()
        tokens = manager.list()
        processes = list()

        tokens.extend([None] * len(self.sentences))

        for idx, sentence in enumerate(self.sentences):
            processes.append(Process(target=word_tokenization, args=(sentence, idx, tokens)))

        for p in processes:
            p.start()
        for p in processes:
            p.join()
        self.tokens = tokens
        Helper.debug("word_tokenization", True, "module_debug")

    def remove_punctuation(self):
        for idx, (token_list, sentence) in enumerate(zip(self.tokens, self.sentences)):
            self.unpunc_sentences.append(
                self.remove_punc(sentence, [t for t in token_list if t["type"] == "Punctuation"])
            )

    def lemmatization(self):
        manager = multiprocessing.Manager()

        lemmas = manager.list()
        lemmas.extend([None] * len(self.sentences))

        p = Process(target=find_lemmas, args=(self.filtered_sentences, lemmas))
        p.start()
        p.join()
        self.lemmas["raw"] = lemmas
        for lemma_sentence in lemmas:
            t_l, s_l, s = self.extract_token_and_pos(lemma_sentence)
            self.lemmas["tuple"].append(t_l)
            self.lemmas["str"].append(s_l)
            self.lemmas["sentence"].append(s)

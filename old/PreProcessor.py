# YAPILACAKLAR
# import pikepdf
# pdf-web-içerik ekleme


import File
from jpype import JClass, getDefaultJVMPath, shutdownJVM, startJVM, JString
import properties
import Helper


class PreProcessor:
    def __init__(self, file_address, on_web, file_type):
        self.file = File.File(file_address, on_web, file_type)

        # print(self.file.on_web, self.file.content_type, end='\n')
        self.sentences = list()
        # self.sentences = self.sentence_tokenize(self.file.raw_text)
        # print("Raw text: \n" + re.sub('(?<![\r\n])(\r?\n|\r)(?![\r\n])', ' ', self.file.raw_text).replace('\n\n', '\n'))
        # print(self.file.raw_text.replace('\n', ' '))
        Helper.init_jvm('-ea', False)
        self.sentences = self.split_into_sentences()
        self.word_tokenize()

    def __del__(self):
        shutdownJVM()

    def split_into_sentences(self):
        if self.sentences:
            return self.sentences
        else:
            turkish_sentence_extractor: JClass = JClass('zemberek.tokenization.TurkishSentenceExtractor')
            extractor: turkish_sentence_extractor = turkish_sentence_extractor.DEFAULT
            print("Sile" + self.file.raw_text)
            sentences = extractor.fromParagraph(self.file.raw_text)
            # s = []
            return [word for word in sentences]
            # for i, word in enumerate(sentences):
            #     s.append(word)
            #     print(f'Sentence {i + 1}: {word}')
            # return s

    def word_tokenize(self):
        turkish_tokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
        token: JClass = JClass('zemberek.tokenization.Token')
        tokenizer: turkish_tokenizer = turkish_tokenizer.DEFAULT

        for sentence in self.sentences:
            token_iterator = tokenizer.getTokenIterator(JString(sentence))
            for token in token_iterator:
                print((
                    f'Token = {token}'
                    f'\n | Content = {token.content}'
                    f'\n | Normalized = {token.normalized}'
                    f'\n | Type = {token.type}'
                    f'\n | Start = {token.start}'
                    f'\n | End = {token.end}\n'
                ))

# data/İnsan Ne İle Yaşar - Lev Nikolayeviç Tolstoy ( PDFDrive.com ).pdf

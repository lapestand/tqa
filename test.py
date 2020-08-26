# # import os
# # from multiprocessing import Process
# # import time
# # from os.path import join
# # from jpype import JClass, getDefaultJVMPath, shutdownJVM, startJVM, JString
# #
# # COUNT = 5000000
# #
# # zemberek_path: str = join('Dependencies', 'Zemberek-Python', 'bin', 'zemberek-full.jar')
# #
# #
# # def test():
# #     try:
# #         startJVM(
# #             getDefaultJVMPath(),
# #             '-ea',
# #             f'-Djava.class.path={zemberek_path}',
# #             convertStrings=False
# #         )
# #
# #         TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
# #         Token: JClass = JClass('zemberek.tokenization.Token')
# #
# #         tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT
# #
# #         inp: str = 'İstanbul\'a, merhaba! Deneme yazısı lo bu.'
# #
# #         print('\nToken Iterator Example:\n')
# #
# #         print(f'Input = {inp}\n')
# #
# #         token_iterator = tokenizer.getTokenIterator(JString(inp))
# #         for token in token_iterator:
# #             print((
# #                 f'Token = {token}'
# #                 f'\n | Content = {token.content}'
# #                 f'\n | Normalized = {token.normalized}'
# #                 f'\n | Type = {token.type}'
# #                 f'\n | Start = {token.start}'
# #                 f'\n | End = {token.end}\n'
# #             ))
# #
# #         print('Default Tokenization Example:\n')
# #
# #         tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT
# #
# #         print(f'Input = {inp}')
# #         for i, token in enumerate(tokenizer.tokenizeToStrings(
# #                 JString(inp)
# #         )):
# #             print(f' | Token String {i} = {token}')
# #
# #         print('\nCustom Tokenization Example:\n')
# #
# #         tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
# #             Token.Type.Punctuation,
# #             Token.Type.NewLine,
# #             Token.Type.SpaceTab
# #         ).build()
# #         inp: str = 'Saat, 12:00'
# #         print(f'Input = {inp}')
# #         for i, token in enumerate(tokenizer.tokenize(JString(inp))):
# #             print(f' | Token {i} = {token}')
# #
# #         shutdownJVM()
# #         print("done")
# #     except:
# #         print("Error")
# #
# #
# # if __name__ == '__main__':
# #     start = time.time()
# #     p_list = list()
# #     p_list.append(Process(target=test))
# #     p_list.append(Process(target=test))
# #
# # for _ in p_list:
# #     _.start()
# #
# # for _ in p_list:
# #     _.join()
# #
# # end = time.time()
# # print("Time taken in seconds -", end - start)
#
# from nltk.corpus import stopwords
#
# from nltk.tokenize import word_tokenize
#
# example_sent = "This is a sample sentence, showing off the stop words filtration."
#
# stop_words = set(stopwords.words('english'))
#
# word_tokens = word_tokenize(example_sent)
#
# filtered_sentence = [w for w in word_tokens if w not in stop_words]
#
# print(filtered_sentence)
#
# filtered_sentence = []
#
# for w in word_tokens:
#     if w not in stop_words:
#         filtered_sentence.append(w)
#
# print(word_tokens)
# print(filtered_sentence)


# import numpy as np
# import cv2
# import pyautogui
#
# # take screenshot using pyautogui
# image = pyautogui.screenshot()
#
# # since the pyautogui takes as a
# # PIL(pillow) and in RGB we need to
# # convert it to numpy array and BGR
# # so we can write it to the disk
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
# # writing it to the disk using opencv
# cv2.imwrite("image1.png", image)
#
#
# import time
# import cv2
# import mss
# import numpy as np
#
# with mss.mss() as sct:
#     # Part of the screen to capture
#     monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
#
#     while True:
#         last_time = time.time()
#
#         # Get raw pixels from the screen, save it to a Numpy array
#         img = np.array(sct.grab(monitor))
#
#         # Display the picture
#         cv2.imshow('frame', img)
#
#         print("fps: {}".format(1 / (time.time() - last_time)))
#
#         # Press "q" to quit
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             cv2.destroyAllWindows()
#             break
# import cv2
# import numpy as np
# import pyautogui
#
# screen_size = tuple(pyautogui.size())
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, (screen_size))
#
# while True:
#     # make a screenshot
#     img = pyautogui.screenshot()
#     # convert these pixels to a proper numpy array to work with OpenCV
#     frame = np.array(img)
#     # convert colors from BGR to RGB
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # write the frame
#     out.write(frame)
#     # show the frame
#     cv2.imshow("screenshot", frame)
#     # if the user clicks q, it exits
#     if cv2.waitKey(1) == ord("q"):
#         break
#
# cv2.destroyAllWindows()
# out.release()
#

# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture('output.avi')
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# import pandas as pd
#
# df = pd.DataFrame(columns=["Ali", "Veli", "49", "Ahmet"])
#
# print(df)
#
# new_row = pd.Series([41, 34, "sdf", 43], index=df.columns.to_numpy().tolist())
# df = df.append(new_row, ignore_index=True)
#
# print(df)
# 
# 
# import fasttext.util
#
# print("Downloading..")
# fasttext.util.download_model('tr', if_exists='ignore')  # Turkish
# ft = fasttext.load_model('cc.tr.300.bin')
#
# print(f"Dimension: {ft.get_dimension()}")
# fasttext.util.reduce_model(ft, 100)
# print(f"Dimension: {ft.get_dimension()}")
#
# test_str = "Merhaba"
# print(f"Word vector shape of {test_str} -> {ft.get_word_vector(test_str).shape}")
# print(ft.get_nearest_neighbors(test_str))

# from multiprocessing import Value
# from ctypes import c_bool
# file_a = Value(c_bool, False)
# file_n = Value(c_bool, False)
# file_c = Value(c_bool, True)
#
# print(file_a)
# print(file_a.value)
# file_a.value = True
# print(file_a)
# print(file_a.value)

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
from src import Helper
from src.Helper import Properties
from src import Helper
from src.pre_processing.PreProcessor import PreProcessor
#
# doc = ["Test yapıyorum. Acaba ne kadar doğru? Emin de değilim, aslında."]
#
# pp = PreProcessor()
# pp.split_into_sentences(doc)
# print(pp.data["raw_data"]["sentences"])
# s = [_.lower() for _ in (pp.data["raw_data"]["sentences"])]
# print([_.lower() for _ in (pp.data["raw_data"]["sentences"])])
#
#
# def word_tokenization(sent):
#     def run(sentence, i, rt_tokens, rt):
#         try:
#             startJVM(
#                 getDefaultJVMPath(),
#                 '-ea',
#                 f'-Djava.class.path={Properties.zemberek_path}',
#                 convertStrings=False
#             )
#
#             tokens = list()
#             TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
#             Token: JClass = JClass('zemberek.tokenization.Token')
#
#             tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT
#
#             inp = sentence
#
#             token_iterator = tokenizer.getTokenIterator(JString(inp))
#             for token in token_iterator:
#                 tokens.append(
#                     {
#                         "token": str(token),
#                         "content": str(token.content),
#                         "normalized": str(token.normalized),
#                         "type": str(token.type),
#                         "start": str(token.start),
#                         "end": str(token.end)
#                     }
#                 )
#             if rt_tokens[i] is None:
#                 rt_tokens[i] = tokens
#             else:
#                 print("\n\n\n\n\nAMAZING ERROR\n\n\n\n\n")
#                 rt[i] = False
#             shutdownJVM()
#         except:
#             rt.value = False
#             Helper.debug("word_tokenization", False, "module_debug")
#
#     manager = multiprocessing.Manager()
#     tokens = manager.list()
#     processes = list()
#     ret_values = manager.list()
#     ret_values.extend([True] * len(sent))
#
#     tokens.extend([None] * len(sent))
#
#     for idx, s1 in enumerate(sent):
#         processes.append(Process(target=run, args=(s1, idx, tokens, ret_values)))
#
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()
#
#     return False in ret_values
#
#
#
# for _ in word_tokenization(s):
#     print(_, end="\n\n\n")

import question_answering

while True:
    question_answering.main()
print("HI")
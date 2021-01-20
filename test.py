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
#
# import question_answering
#
# while True:
#     question_answering.main()
# print("HI")

# print(__doc__)
#
#
# # Code source: Gaël Varoquaux
# #              Andreas Müller
# # Modified for documentation by Jaques Grobler
# # License: BSD 3 clause
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_moons, make_circles, make_classification
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#
# h = .02  # step size in the mesh
#
# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]
#
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
#
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)
#
# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]
#
# figure = plt.figure(figsize=(27, 9))
# i = 1
# # iterate over datasets
# for ds_cnt, ds in enumerate(datasets):
#     # preprocess dataset, split into training and test part
#     X, y = ds
#     X = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = \
#         train_test_split(X, y, test_size=.4, random_state=42)
#
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#
#     # just plot the dataset first
#     cm = plt.cm.RdBu
#     cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#     ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#     if ds_cnt == 0:
#         ax.set_title("Input data")
#     # Plot the training points
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#                edgecolors='k')
#     # Plot the testing points
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
#                edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xticks(())
#     ax.set_yticks(())
#     i += 1
#
#     # iterate over classifiers
#     for name, clf in zip(names, classifiers):
#         ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)
#
#         # Plot the decision boundary. For that, we will assign a color to each
#         # point in the mesh [x_min, x_max]x[y_min, y_max].
#         if hasattr(clf, "decision_function"):
#             Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#         else:
#             Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
#         # Put the result into a color plot
#         Z = Z.reshape(xx.shape)
#         ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
#
#         # Plot the training points
#         ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#                    edgecolors='k')
#         # Plot the testing points
#         ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
#                    edgecolors='k', alpha=0.6)
#
#         ax.set_xlim(xx.min(), xx.max())
#         ax.set_ylim(yy.min(), yy.max())
#         ax.set_xticks(())
#         ax.set_yticks(())
#         if ds_cnt == 0:
#             ax.set_title(name)
#         ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#                 size=15, horizontalalignment='right')
#         i += 1
#
# plt.tight_layout()
# plt.show()
#
# # !/usr/bin/env python
# import socket
# import subprocess
# import sys
# from datetime import datetime
#
# # Clear the screen
# subprocess.call('clear', shell=True)
#
# # Ask for input
# remoteServer = input("Enter a remote host to scan: ")
# remoteServerIP = socket.gethostbyname(remoteServer)
#
# # Print a nice banner with information on which host we are about to scan
# print("-" * 60)
# print("Please wait, scanning remote host", remoteServerIP)
# print("-" * 60)
#
# # Check what time the scan started
# t1 = datetime.now()
#
# # Using the range function to specify ports (here it will scans all ports between 1 and 1024)
#
# # We also put in some error handling for catching errors
#
# try:
#     for port in range(1, 1025):
#         print(f"port: {port}", end="\n\n")
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         result = sock.connect_ex((remoteServerIP, port))
#         if result == 0:
#             print("Port {}: 	 Open".format(port))
#         else:
#             print(f"Port {range} Close. Result: {result}")
#         sock.close()
#
# except KeyboardInterrupt:
#     print("You pressed Ctrl+C")
#     sys.exit()
#
# except socket.gaierror:
#     print('Hostname could not be resolved. Exiting')
#     sys.exit()
#
# except socket.error:
#     print("Couldn't connect to server")
#     sys.exit()
#
# # Checking the time again
# t2 = datetime.now()
#
# # Calculates the difference of time, to see how long it took to run the script
# total = t2 - t1
#
# # Printing the information to screen
# print('Scanning Completed in: ', total)
from os import system

system("clear")

# import json
#
# file_loc = 'Dependencies/turkish-nlp-qa-dataset-master/combined.json'
# question_count, markless = 0, 0
# with open(file_loc, 'r') as f:
#     distros_dic = json.load(f)
# for distro in distros_dic["data"]:
#     for p in distro["paragraphs"]:
#         for q in p["qas"]:
#             question_count += 1
#             if "?" not in q["question"]:
#                 markless += 1
#                 print(q["question"])
# print(f"\n\nQuestion count: {question_count}")
# print(f"\n\nMarkless count: {markless}")

# from src.qanswering.analyser.SemanticAnalyzer import SwoExtractor
#
# swo = SwoExtractor()
#
# if swo.load_questions():
#     print(len(swo.questions))
#     swo.tokenize_questions(part_size=20)
# #     swo.tokenize_questions(p_mode=True)

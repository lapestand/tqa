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

a = "aliveli"

print(list(a))

print([a])

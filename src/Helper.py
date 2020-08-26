import os
import shutil
import http.client as httplib
from os.path import join
from urllib.parse import urlparse


# import fasttext.util


class Properties:
    DEBUG = True

    supported_file_types = [
        "pdf", "html", "txt"
    ]

    error_messages = {
        "unsupported_file_type": "\nError: Unsupported file type! Please enter a valid file type!"
    }

    referance_url = "www.google.com"

    zemberek_path: str = join('Dependencies', 'Zemberek-Python', 'bin', 'zemberek-full.jar')

    error_file_name = "Errors.txt"

    analyzer_mode = {"syntactic": True, "semantic": False}

    distance_mode = {"euclidian_distance": False, "cosine_similarity": True}

    def error_message(self, line, func):
        return "Error on %s line - function name: %s", line, func

    def __init__(self):
        pass

    def __del__(self):
        pass


def rm_r(self, path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def chunk_size(self):
    return 8192


def debug(module_name, detail, mode):
    situation_details = ["starting...", "started!", "done!"]
    if Properties.DEBUG:
        if mode == "module_debug":
            print('\t' + module_name + ' ' + ('✔' if detail else '✗'))
        elif mode == "situation":
            module_name += " - "
            detail = situation_details[detail]
            print(module_name + detail)
        elif mode == "info":
            print("\t\t" + module_name + ': ' + detail)

# def get_started():
#     fasttext.util.download_model('tr', if_exists='ignore')

# def mute()
import os
import shutil
from os.path import join

ZEMBEREK_PATH: str = join('Dependencies', 'Zemberek-Python', 'bin', 'zemberek-full.jar')


def error_file_name():
    return "Errors.txt"


def rm_r(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def chunk_size():
    return 8192


supported_file_types = [
    "pdf", "html", "txt"
]

error_messages = {
    "unsupported_file_type": "\nError: Unsupported file type! Please enter a valid file type!"
}



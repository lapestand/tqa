import pdftotext
import requests
import os
import properties
import shutil
from pathlib import Path
import Helper


class File:
    def __init__(self, link_to_file, on_web, file_type):
        self.on_web = on_web
        self.file_data_dir = os.getcwd() + "/data/file_data"

        self.create_repository()

        self.file_name = link_to_file.split('/')[-1] if self.on_web else os.path.basename(link_to_file)
        print("File name is: " + self.file_name)
        self.file_type = file_type if file_type else self.get_content_type()

        self.add2project(link_to_file)
        self.raw_text = None
        self.raw_text = self.get_raw_text()

        # self.response = None
        # self.file_data_dir = "data/file_class"
        # os.mkdir(self.file_data_dir)
        # self.on_web = self.on_web_(file_address)
        # self.file_name = file_address.split('/')[-1] if self.on_web else os.path.basename(file_address)
        # self.content_type = self.content_type_(file_address)
        # self.file_name = self.file_data_dir + self.file_name + self.content_type
        # self.content = self.content_(file_address)

    def __del__(self):
        shutil.rmtree(self.file_data_dir, ignore_errors=True)

    def create_repository(self):
        Path(self.file_data_dir).mkdir(parents=True, exist_ok=True)

    def get_content_type(self):
        if self.file_type is not None:
            return self.file_type
        else:
            content_type = self.file_name.split('.')[-1].lower()

            print("Content type is: " + content_type)

            if "application/pdf" in content_type:
                return "pdf"
            elif "text/html" in content_type:
                return "html"
            elif "pdf" in content_type:
                return "pdf"
            elif "txt" in content_type:
                return "txt"
            else:
                return "UNKNOWN TYPE"

    def get_raw_text(self):
        if self.raw_text is not None:
            return self.raw_text
        else:
            if self.file_type == "pdf":
                with open(self.file_data_dir + '/' + self.file_name, "rb") as f:
                    pdf = pdftotext.PDF(f)
                return " ".join(pdf) if pdf else Helper.get_text_using_ocr(pdf)

            elif self.file_type == "txt":
                with open(self.file_data_dir + '/' + self.file_name, "r") as f:
                    return f.readlines()[0]

            elif self.file_type == "html":
                with open(self.file_data_dir + '/' + self.file_name, "r") as f:
                    return " ".join(f.readlines())

    def add2project(self, src):
        if self.on_web:
            try:
                # html = self.download(src)
                # if self.file_type == "html":
                #     Helper.html_pruning(html)
                self.download(src)
            except:
                print("Error while downloading!")
        else:
            try:
                self.copy(src)
            except:
                print("Error while copying!")

    def download(self, src):
        # print(self.file_data_dir)
        r = requests.get(src, allow_redirects=True)
        with open(self.file_data_dir + '/' + self.file_name, "wb") as f:
            f.write(r.content)
        return r.content
        # with open(self.file_data_dir + '/' + self.file_name, "wb") as pdf:
        #     for chunk in r.iter_content(chunk_size=Properties.chunk_size()):
        #         if chunk:
        #             pdf.write(chunk)

    def copy(self, src):
        # print("add2project[DEBUG] - Src: " + src + "\nDst: " + self.file_data_dir + '/' + self.file_name)
        if os.path.isabs(src):
            shutil.copyfile(src, self.file_data_dir + '/' + self.file_name)
        else:
            shutil.copyfile(os.getcwd() + '/' + src, self.file_data_dir + '/' + self.file_name)

# 00000010100000110010111001001110010101

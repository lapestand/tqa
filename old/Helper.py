import jpype
from jpype import JClass, getDefaultJVMPath, shutdownJVM, startJVM, JString
import properties
from io import StringIO
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def init_jvm(arg, convert_str):
    if jpype.isJVMStarted():
        return
    startJVM(
        getDefaultJVMPath(),
        arg,
        f'-Djava.class.path={properties.ZEMBEREK_PATH}',
        convertStrings=convert_str
    )


def html_pruning(html):
    return None


def get_text_using_ocr(pdf):
    return pdf

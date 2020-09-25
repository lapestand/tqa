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

    analyzer_mode = {"Syntactic": True, "RuleBased_hybrid": False, "ANN": False}

    mode = {"TfIdf+PoS": False}

    distance_mode = {"euclidian_distance": True, "cosine_similarity": False}

    questionPOS = {
        "Num": [('kaç', 'Adj'), ('kaç', 'Verb'), ('kaçta', 'Adv'), ('kaçıncı', 'Adj'), ('kadar', 'Postp'), ],
    }

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

"""
Baltimore’da büyüdüm. Çocukluğumdan beri şahsen travma ve zihinsel sağlık sorunlarıyla mücadele ettim. Tepki vermek, en büyük savunma mekanizmalarımdan biri oldu. Babam, hayatımın çoğunluğunu hapishaneye girip çıkarak geçiren bir eroin bağımlısıydı, annemi beni yalnız başına büyütmek zorunda bıraktı.

Çocukların görmemesi gereken şeyleri gördüm; yetişkinlerin yaşamadığı deneyimleri yaşadım. 14 yaşımdayken akut anksiyete ve orta şiddette depresyona yakalanmıştım. Daha sonra 26 yaşında Jeneralize Anksiyete ve Panik Bozukluğu ve orta-ağır majör depresif bozukluk tanısı kondu.

Teşhisimden hemen sonra, doktorum tarafından hastalığımın şiddetli olduğu ve ilaç tedavisinin tek etkili seçenek olduğu söylendi ve ilaçlar başlandı.

Açıklama projenin ortaklarından Rus enerji devi Gazprom dan geldi. Yıllık 63 milyar metreküp enerji.

ilk günündeki 20 yarış heyecanlıydı, 109 puan toplayan Türkiye, 12 ülke arasında 9. oldu ve yarış tamamlandı.

Cortananın yeni işletim sistemi Windows 10 un önemli bir parçası olduğunu belirten Microsoft ; Google Android ve iOS cihazlarındaki Dijital.

Teknoloji devi Google, Android in MMM sürümüyle birlikte bir çok sistemsel hatasının düzeltileceğini.

Siroz hastalığı ile ilgili detaylara dikkat çekerek, sağlıklı bir karaciğere sahip olmak hastalık için.

Hastalık çoğu kez yıllarca doğru tanı konmaması veya ciddiye alınmaması sebebi ile kısırlaştırıcı etki yapabiliyor, kronik ağrı.

Ahmet Razgatlıoğlu bir sporcuydu.

ilk 4 etaptan galibiyetle ayrılan 18 yaşındaki Razgatlıoğlu, Almanya daki yarışta 3. sırayı alarak.

Helal gıda pazarı sanki 860 milyar doların üzerinde.
"""

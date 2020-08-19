import src.Helper
from src.Helper import Properties
import time


def main():
    program_time = {
        "main_program": dict(),
        "input_validation": dict(),
        "data_receiving": dict(),
        "pre_processing": dict(),
        "modelling": dict(),
        "q-a": dict()
    }
    program_time["main_program"]["start"] = time.time()
    ###########################
    """ INPUT VALIDATION """  #
    program_time["input_validation"]["start"] = time.time()
    from src.data_receiving.InputValidator import InputValidator
    #####################################################################################

    # file_address = input("Enter file: ")
    # on_web = input("Is file on the web(yes/no): ").lower() == 'yes'
    # file_type = input(
    #     "Please Enter file type " + "Supported file types are -> " + ", ".join(
    #         Properties.supported_file_types) + " : ").lower()
    # file_address = "data/test.txt"
    # question = input("Enter question")
    questions = ["Nerede büyüdün?", "Neyle mücadele ettin?", "Baban nasıl biriydi?",
                 "Hiç psikolojik bir rahatsızlık geçirdin mi?"]
    file_type = "txt"
    on_web = False
    file_address = "data/test.txt"

    input_validator = InputValidator()
    input_validator.validation((file_address, on_web, file_type))

    # print(input_validator.is_okey)
    if not input_validator.is_okey:
        # print("ERROR\nProblems:")
        # print([cr for cr, st in input_validator.criteria.items() if not st])
        return
    ###
    program_time["input_validation"]["end"] = time.time()
    #####################################################################################

    #########################
    """ DATA RECEIVING """  #
    program_time["data_receiving"]["start"] = time.time()
    from src.data_receiving.DataReceiver import DataReceiver
    #####################################################################################
    receiver = DataReceiver(file_address, on_web, file_type)
    data_inf = receiver.receive()
    if not data_inf:
        print("empty")
        return
    raw_data, file_name = data_inf
    ###
    program_time["data_receiving"]["end"] = time.time()
    #####################################################################################

    #########################
    """ PRE-PROCESSING """  #
    program_time["pre_processing"]["start"] = time.time()
    from src.pre_processing.PreProcessor import PreProcessor
    #####################################################################################
    pre_processor = PreProcessor(raw_data, file_name, file_type)
    pre_processor.pre_process()

    # for s, t, sentence in zip(pre_processor.lemmas["str"], pre_processor.lemmas["tuple"], pre_processor.lemmas["sentence"]):
    #     print(f"\n\n\nString = {s}\n\nTuple = {t}\n\nSentence = {sentence}\n")
    ###
    program_time["pre_processing"]["end"] = time.time()
    #####################################################################################

    #########################
    """    MODELLING   """  #
    program_time["modelling"]["start"] = time.time()
    from src.modelling.Model import Model
    #####################################################################################
    model = Model(pre_processor.lemmas["tuple"])
    # document_term_matrix = model.document_term_matrix_tfidf()
    # print(document_term_matrix)
    ###
    program_time["modelling"]["end"] = time.time()
    #####################################################################################

    #########################
    """       Q-A      """  #
    program_time["q-a"]["start"] = time.time()
    from src.qanswering.question_analyzer import Analyser
    #####################################################################################
    analyzer = Analyser(model, pre_processor.untouched_sentences)
    parsed_questions = analyzer.parse_questions(questions)
    for idx, question in enumerate(questions):
        print(f"\n\nQuestion-{idx + 1}: {question}\n"
              f"Answer-{idx + 1}: {analyzer.answer(question, idx)}\n\n\n")
    ###
    program_time["q-a"]["end"] = time.time()
    #####################################################################################

    program_time["main_program"]["end"] = time.time()

    print("\n\n\n")
    for key, value in program_time.items():
        print(f"Elapsed time for {key} is ->\t{program_time[key]['end'] - program_time[key]['start']}", end='\n')


if __name__ == '__main__':
    main()

"""
ilk_repolar = ilk hesaptaki repo isimleri
ikinci_repolar = ikinci hesaptaki repo isimleri

Bunları küme gibi düşünürsek
olmayan repolar = ilk_repolar - ikinci_repolar (farkı)

ilk_repolar ∩ ikinci_repolar 
"""
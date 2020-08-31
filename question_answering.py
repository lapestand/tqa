from src import Helper
from src.Helper import Properties
import time


def main():
    # Helper.get_started()
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
    questions = ["Razgatlığoğlu ilk etaptan kaç galibet kazandı?", "Nerede büyüdün?", "Neyle mücadele ettin?",
                 "Baban nasıl biriydi?",
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
    pre_processor = PreProcessor()
    pre_processor.pre_process(raw_data, file_type)
    # pre_processor.pre_process(raw_data, file_type, remove_stop_words=True)

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
    model = Model(pre_processor.data["p_data"]["pos"]["tuple"])
    document_term_matrix = model.document_term_matrix_tfidf()
    if Properties.DEBUG:
        print(document_term_matrix)
    ###
    program_time["modelling"]["end"] = time.time()
    #####################################################################################

    #########################
    """       Q-A      """  #
    program_time["q-a"]["start"] = time.time()
    from src.qanswering.question_analyzer import Analyser
    #####################################################################################
    analyzer = Analyser(model, pre_processor.data["raw_data"]["sentences"],
                        pre_processor.data["p_data"]["pos"]["tuple"])
    # print(analyzer.matrix_columns, end="\n\n\n")
    for idx, question in enumerate(questions):
        print(f"\n\nQuestion-{idx + 1}: {question}\n"
              f"Answer-{idx + 1}: {analyzer.answer(question, idx)}\n\n\n")
        break

    ###
    program_time["q-a"]["end"] = time.time()
    #####################################################################################

    program_time["main_program"]["end"] = time.time()

    print("\n\n\n")
    for key, value in program_time.items():
        print(f"Elapsed time for {key} is ->\t{program_time[key]['end'] - program_time[key]['start']}", end='\n')


if __name__ == '__main__':
    main()

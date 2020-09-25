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
    method = "TfIdf+PoS"

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

    from src.Mastermind import Mastermind
    question = questions[0]
    mind = Mastermind(method)
    data, model = mind.read(raw_data, file_type)
    answer = mind.find_answer(to=question, looking=(
        model, data["raw_data"]["sentences"], data["p_data"]["pos"]["tuple"]
    ))

    print(f"\n\nAnswer:\n\n{answer}")

    program_time["main_program"]["end"] = time.time()

    # print("\n\n\n")
    # for key, value in program_time.items():
    #     print(f"Elapsed time for {key} is ->\t{program_time[key]['end'] - program_time[key]['start']}", end='\n')


if __name__ == '__main__':
    main()

from src.Helper import Properties


def read(raw_data, file_type):
    #########################
    """ PRE-PROCESSING """  #
    # program_time["pre_processing"]["start"] = time.time()
    from src.pre_processing.PreProcessor import PreProcessor
    #####################################################################################
    pre_processor = PreProcessor()
    pre_processor.pre_process(raw_data, file_type)
    # pre_processor.pre_process(raw_data, file_type, remove_stop_words=True)

    # for s, t, sentence in zip(pre_processor.lemmas["str"], pre_processor.lemmas["tuple"], pre_processor.lemmas["sentence"]):
    #     print(f"\n\n\nString = {s}\n\nTuple = {t}\n\nSentence = {sentence}\n")
    ###
    # program_time["pre_processing"]["end"] = time.time()
    #####################################################################################

    #########################
    """    MODELLING   """  #
    # program_time["modelling"]["start"] = time.time()
    from src.modelling.Model import Model
    #####################################################################################
    model = Model(pre_processor.data["p_data"]["pos"]["tuple"])
    document_term_matrix = model.document_term_matrix_tfidf()
    if Properties.DEBUG:
        print(document_term_matrix)
    ###
    # program_time["modelling"]["end"] = time.time()
    #####################################################################################

    return pre_processor.data, model


def answer(to, looking):
    from src.qanswering.Analyzer import Analyser

    analyzer = Analyser(looking[0], looking[1], looking[2])
    return analyzer.answer(to)

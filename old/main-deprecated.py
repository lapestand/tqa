import PreProcessor
import properties


def main():
    # pre_processing()
    # file_name = input("Enter file: ")
    # on_web = True if input("Is file on the web(yes/no)?").lower() == 'yes' else False
    # file_type = input(
    #     "Please Enter file type " + "Supported file types are -> " + ", ".join(properties.supported_file_types) + " : ").lower()
    #
    # if file_type not in properties.supported_file_types:
    #     print(properties.error_messages["unsupported_file_type"])
    #     return -1

    processed_data = PreProcessor.PreProcessor(file_name, on_web, file_type)
    print(processed_data.sentences)
    properties = vars(processed_data)
    # print(processed_data.sentences)
    # for item in properties:
    #     print(item, ':', properties[item])


if __name__ == '__main__':
    main()

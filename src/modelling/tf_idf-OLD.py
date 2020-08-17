import pandas as pd
import math

TF = 0
IDF = False


def info():
    tf_mode = str("TERM" if TF == 0 else ("LOG" if TF == 1 else "AUGMENTED")) + " FREQUENCY"
    idf_mode = str("SMOOTHED " if IDF else "") + "IDF"
    print(f"\n\n\ttf-idf mode -->\t\t"
          f"\t\t\t[{tf_mode}|{idf_mode}]\n")


def find_uniques(bags_of_tokens):
    unique_list = list()
    for bag in bags_of_tokens:
        for token in bag:
            if token not in unique_list:
                unique_list.append(token)
    return unique_list


def sum_of(num_of_tokens_of_docs, tokens):
    sum_dict = dict.fromkeys(tokens, 0)
    for doc in num_of_tokens_of_docs:
        for key, value in doc.items():
            sum_dict[key] += value
    return sum_dict


def tf(token_dict, bag_of_words, normalization=TF):
    tf_dict = dict()
    bag_of_words_count = len(bag_of_words)
    relevant_token_dict = dict((k, v) for k, v in token_dict.items() if v > 0)

    if normalization == 0:
        for token, count in relevant_token_dict.items():
            tf_dict[token] = count / float(bag_of_words_count)
    elif normalization == 1:
        for token, count in relevant_token_dict.items():
            tf_dict[token] = math.log(1 + count)
    elif normalization == 2:
        k = 0.5
        for token, count in relevant_token_dict.items():
            tf_dict[token] = k + ((1 - k) * (count / max(relevant_token_dict.values())))
    return tf_dict


def idf(documents, bags, smooth=IDF):
    n = len(documents)
    idf_dict = sum_of(documents, documents[0].keys())
    if smooth:
        for token, count in idf_dict.items():
            idf_dict[token] = math.log((1 + n) / (1 + float(count))) + 1
    else:
        for token, count in idf_dict.items():
            idf_dict[token] = math.log(n / float(count))
    return idf_dict


def tf_idf(tf_, idf_):
    tf_idf_dict = dict()
    for token, count in tf_.items():
        tf_idf_dict[token] = count * idf_[token]
    return tf_idf_dict


def document_term_matrix(bags_of_tokens):
    info()
    num_of_tokens_of_docs = list()
    unique_tokens = list(set(token for bag in bags_of_tokens for token in bag))

    for bag in bags_of_tokens:
        num_of_tokens_of_docs.append(dict.fromkeys(unique_tokens, 0))
        for token in bag:
            num_of_tokens_of_docs[-1][token] += 1

    all_tfs = list()

    for idx, doc in enumerate(num_of_tokens_of_docs):
        all_tfs.append(tf(doc, bags_of_tokens[idx]))
    idf_ = idf(num_of_tokens_of_docs, bags_of_tokens)

    tf_idfs = list()
    for tf_ in all_tfs:
        tf_idfs.append(tf_idf(tf_, idf_))

    return pd.DataFrame(tf_idfs).fillna(0.0), idf_

    # vectorizer = TfidfVectorizer()
    #
    # vectors = vectorizer.fit_transform(bags_of_tokens)
    #
    # feature_names = vectorizer.get_feature_names()
    #
    # dense = vectors.todense()
    # dense_list = dense.tolist()
    #
    # df = pd.DataFrame(dense_list, columns=feature_names)

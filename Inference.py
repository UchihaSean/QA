# This Python file uses the following encoding: utf-8
from TFIDF import TFIDF
from CNN_train import CNN
from LM import LM
import Data
import time
import random


def main():
    random.seed(12345)
    retrieval_data_start_time = time.clock()
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")
    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(pred_questions)

    top_k = 3

    print("Retrieval Data Finished")

    retrieval_data_end_time = time.clock()
    print("Retrieval Data cost %f" % (retrieval_data_end_time - retrieval_data_start_time))

    response_start_time = time.clock()

    lm = LM(top_k, questions, pred_questions, answers, pred_answers, word_sentence_dict)
    tfidf = TFIDF(top_k * 40, questions, pred_questions, answers, pred_answers, word_sentence_dict)
    cnn = CNN(top_k, questions, pred_questions, answers, pred_answers, word_sentence_dict, isTrain=False)

    qs_input = "这个笔记本好用么"
    print("Question : %s" % qs_input)

    _, lm_response = lm.ask_response(qs_input)
    tfidf_response_id, tfidf_response = tfidf.ask_response(qs_input)
    cnn_response = cnn.ask_response(qs_input, tfidf_response_id)

    for i in range(top_k):
        print("LM response %d: %s" % (i + 1, lm_response[i]))
    for i in range(top_k):
        print("TFIDF response %d: %s" % (i + 1, tfidf_response[i]))
    for i in range(top_k):
        print("CNN response %d: %s" % (i + 1, cnn_response[i]))

    print("Response Finished")

    response_end_time = time.clock()
    print("Response cost %f" % (response_end_time - response_start_time))


if __name__ == "__main__":
    main()

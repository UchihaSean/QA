# This Python file uses the following encoding: utf-8
from TFIDF import TFIDF
from CNN_train import CNN
from LM import LM
import Data
import time
import random
import numpy as np


def ask_question(qs_input, top_k):
    """
    Ask one question and generate response for tfidf, lm and cnn
    """

    print("Question : %s" % qs_input)
    print("Top k : : %d" % top_k)

    random.seed(12345)
    retrieval_data_start_time = time.clock()
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")
    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(pred_questions)

    print("Retrieval Data Finished")

    retrieval_data_end_time = time.clock()
    print("Retrieval Data cost %f" % (retrieval_data_end_time - retrieval_data_start_time))

    response_start_time = time.clock()

    lm = LM(questions, pred_questions, answers, pred_answers, word_sentence_dict)
    tfidf = TFIDF(questions, pred_questions, answers, pred_answers, word_sentence_dict)
    cnn = CNN(questions, pred_questions, answers, pred_answers, word_sentence_dict, isTrain=False)

    _, lm_response = lm.ask_response(qs_input, top_k=top_k)
    tfidf_response_id, tfidf_response = tfidf.ask_response(qs_input, top_k=top_k * 10)
    cnn_response = cnn.ask_response(qs_input, top_k, tfidf_response_id)

    for i in range(top_k):
        print("LM response %d: %s" % (i + 1, lm_response[i]))
    for i in range(top_k):
        print("TFIDF response %d: %s" % (i + 1, tfidf_response[i]))
    for i in range(top_k):
        print("CNN response %d: %s" % (i + 1, cnn_response[i]))

    print("Response Finished")

    response_end_time = time.clock()
    print("Response cost %f" % (response_end_time - response_start_time))


def cnn_output(output_num, top_k):
    """
    Generate cnn outputs
    """
    random.seed(12345)
    retrieval_data_start_time = time.clock()
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")
    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(pred_questions)

    print("Retrieval Data Finished")

    retrieval_data_end_time = time.clock()
    print("Retrieval Data cost %f" % (retrieval_data_end_time - retrieval_data_start_time))

    response_start_time = time.clock()

    tfidf = TFIDF(questions, pred_questions, answers, pred_answers, word_sentence_dict)
    cnn = CNN(questions, pred_questions, answers, pred_answers, word_sentence_dict, isTrain=False)

    output = open("cnn.txt", "w")
    for i in range(output_num):
        qs_index = int(random.random() * len(questions))
        qs_input = questions[qs_index]
        tfidf_response_id, tfidf_response = tfidf.ask_response(qs_input, top_k * 10)
        cnn_response = cnn.ask_response(qs_input, top_k, tfidf_response_id)


def main():
    qs_input = "屏幕分辨率多少"
    top_k = 3
    ask_question(qs_input, top_k)


if __name__ == "__main__":
    main()

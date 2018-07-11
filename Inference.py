# This Python file uses the following encoding: utf-8
from TFIDF import TFIDF
from CNN_train import CNN
from LM import LM
import Data
import time


def main():
    retrieval_data_start_time = time.clock()
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")
    top_k = 3

    print("Retrieval Data Finished")

    retrieval_data_end_time = time.clock()
    print("Retrieval Data cost %f" % (retrieval_data_end_time - retrieval_data_start_time))

    response_start_time = time.clock()

    lm = LM(top_k, questions, pred_questions, answers, pred_answers)
    tfidf = TFIDF(top_k*40, questions, pred_questions, answers, pred_answers)
    cnn = CNN(top_k, questions,pred_questions, answers, pred_answers)

    qs_input = "有什么电脑推荐么"
    _, lm_response = lm.ask_response(qs_input)
    tfidf_response_id, tfidf_response = tfidf.ask_response(qs_input)
    cnn_response = cnn.ask_response(qs_input, tfidf_response_id)

    print("Question : %s" % qs_input)
    for i in range(top_k):
        print("LM response %d: %s" % (i + 1, lm_response[i]))
    for i in range(top_k):
        print("TFIDF response %d: %s" % (i + 1, tfidf_response[i]))
    for i in range(top_k):
        print("CNN response %d: %s" %(i+1, cnn_response[i]))

    print("Response Finished")

    response_end_time = time.clock()
    print("Response cost %f" % (response_end_time - response_start_time))


if __name__ == "__main__":
    main()

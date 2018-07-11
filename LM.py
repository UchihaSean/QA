# This Python file uses the following encoding: utf-8
import Data
import numpy as np
import heapq
import random


class LM:
    def __init__(self, top_k, questions=None, pred_questions=None, answers=None, pred_answers=None, word_sentence_dict = None):
        # Read Preprocessed Data
        if questions == None:
            self.questions, self.pred_questions, self.answers, self.pred_answers = Data.read_pred_data(
                "Data/pred_QA-pair.csv")
            # Build word --> sentence dictionary
            self.word_sentence_dict = Data.generate_word_sentence_dict(self.pred_questions)

        else:
            self.questions, self.pred_questions, self.answers, self.pred_answers, self.word_sentence_dict = questions, pred_questions, answers, pred_answers, word_sentence_dict

        self.top_k = top_k

    def ask_response(self, question):
        """
        :param question: input a question
        :return: top k id and response
        """
        pred_q = Data.preprocessing([question.decode("utf-8")])

        top = []

        # Generate sentence id set which include at least one same word
        sentence_id_set = set()
        for j in range(len(pred_q[0])):
            if pred_q[0][j] in self.word_sentence_dict:
                sentence_id_set.update(self.word_sentence_dict[pred_q[0][j]])

        # Generate cosine similarity score
        for j in sentence_id_set:
            score = LM_similarity(pred_q[0], self.pred_questions[j], len(self.pred_questions[j]))
            heapq.heappush(top, (-score, str(j)))

        # print("Question: %s" % question)

        response = []
        response_id = []

        # Generate Top K
        for j in range(min(self.top_k, len(top))):
            item = int(heapq.heappop(top)[1])
            # print("Similar %d: %s" % (j + 1, self.questions[item]))
            # print("LM Response %d: %s" % (j + 1, self.answers[item]))
            response.append(self.answers[item])
            response_id.append(item)

        # print("")

        return response_id, response


def LM_similarity(query, document, doc_len):
    """
    Calculate similarity score for query and document based on LM with Lamplace
    """
    score = 0.0

    # Document dictionary
    doc_dict = {}
    for word in document:
        if word not in doc_dict:
            doc_dict[word] = 1
        else:
            doc_dict[word] += 1

    # Language model
    for word in query:
        if word in doc_dict:
            score += np.log(1 + (doc_dict[word] + 0.0) / doc_len)
    return score


def main():
    # Read Preprocessed Data
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")

    pair = list(zip(questions, pred_questions, answers, pred_answers))
    random.shuffle(pair)
    questions, pred_questions, answers, pred_answers = zip(*pair)

    # Split Data
    split_ratio = 0.7
    split_len = int(len(questions) * split_ratio)
    train_questions = questions[:split_len]
    train_pred_questions = pred_questions[:split_len]
    train_answers = answers[:split_len]
    train_pred_answers = pred_answers[:split_len]
    test_questions = questions[split_len:]
    test_pred_questions = pred_questions[split_len:]
    test_answers = answers[split_len:]
    test_pred_answers = pred_answers[split_len:]

    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(train_pred_questions)

    # Choose the Top K similar ones
    top_k = 5
    output = open("Data/LM.txt", 'w')
    for i in range(len(test_pred_questions)):
        top = []

        # Generate sentence id set which include at least one same word
        sentence_id_set = set()
        for j in range(len(test_pred_questions[i])):
            if test_pred_questions[i][j] in word_sentence_dict:
                sentence_id_set.update(word_sentence_dict[test_pred_questions[i][j]])

        # Generate LM similarity Score
        for j in sentence_id_set:
            score = LM_similarity(test_pred_questions[i], train_pred_questions[j], len(train_pred_questions[j]))
            heapq.heappush(top, (-score, str(j)))

        output.write("Question: " + test_questions[i].encode("utf-8") + "\n")
        output.write("Ground Truth: " + test_answers[i].encode("utf-8") + "\n")

        # Generate Top K
        for j in range(min(top_k, len(top))):
            item = int(heapq.heappop(top)[1])
            output.write("Our similar " + str(j + 1) + ": " + train_questions[item].encode("utf-8") + "\n")
            output.write("Our reply " + str(j + 1) + ": " + train_answers[item].encode("utf-8") + "\n")
        output.write("\n")

    output.close()


if __name__ == "__main__":
    # main()
    lm = LM()
    lm.ask_response("有什么好的电脑么")

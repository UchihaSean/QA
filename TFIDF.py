# This Python file uses the following encoding: utf-8
import Data
import heapq
import numpy as np
import random

class TFIDF:
    def __init__(self):
        # Read Preprocessed Data
        self.quetions, self.pred_questions, self.answers, self.pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")

        pair = list(zip(self.quetions, self.pred_questions, self.answers, self.pred_answers))
        random.shuffle(pair)
        self.quetions, self.pred_questions, self.answers, self.pred_answers = zip(*pair)

        # Calculate TF-IDF
        self.idf_dict = generate_idf_dict(self.pred_questions)
        self.tf_idf_pred_questions = generate_tf_idf_list(self.pred_questions, self.idf_dict)

        # Build word --> sentence dictionary
        self.word_sentence_dict = generate_word_sentence_dict(self.pred_questions)

    def ask_response(self, question):
        pred_q = Data.preprocessing([question.decode("utf-8")])
        tf_idf_pred_q = generate_tf_idf_list(pred_q, self.idf_dict)

        top_k = 5
        top = []

        # Generate sentence id set which include at least one same word
        sentence_id_set = set()
        for j in range(len(pred_q[0])):
            if pred_q[0][j] in self.word_sentence_dict:
                sentence_id_set.update(self.word_sentence_dict[pred_q[0][j]])

        # Generate cosine similarity score
        for j in sentence_id_set:
            score = cosine_similarity(tf_idf_pred_q[0], self.tf_idf_pred_questions[j])
            heapq.heappush(top, (-score, str(j)))

        print("Question: %s"% question)


        # Generate Top K
        for j in range(min(top_k, len(top))):
            item = int(heapq.heappop(top)[1])
            # print("Similar %d: %s" % (j + 1, self.quetions[item]))
            print("Response %d: %s" % (j+1,self.answers[item]))

        print("")





def generate_idf_dict(word_list):
    """
    Generate word dictionary based on train data
    """
    dict = {}
    for i in range(len(word_list)):
        flag = set()
        for j in range(len(word_list[i])):
            if word_list[i][j] in flag: continue
            if word_list[i][j] not in dict:
                dict[word_list[i][j]] = 1
            else:
                dict[word_list[i][j]] += 1
            flag.add(word_list[i][j])

    return dict


def generate_tf_idf_list(sentences, idf_dict):
    """
    Generate tf-idf for each word in each sentence
    """
    tf_idf = []
    for sentence in sentences:
        dict = {}
        # Get term frequency
        for word in sentence:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

        # Calculate TF-IDF
        for word in dict:
            if word in idf_dict:
                dict[word] = (1 + np.log(dict[word])) * np.log(len(idf_dict) / (idf_dict[word] + 0.0))
            else:
                dict[word] = 0
        tf_idf.append(dict)

    return tf_idf


def generate_word_sentence_dict(sentences):
    """
    Build word --> sentence id dictionary
    """
    word_sentence_dict = {}
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] in word_sentence_dict:
                word_sentence_dict[sentences[i][j]].add(i)
            else:
                word_sentence_dict[sentences[i][j]] = {i}
    return word_sentence_dict


def cosine_similarity(dict_x, dict_y):
    """
    Calculate Cosine similarity
    """

    def multiply(dict_u, dict_v):
        """
        Multiply dictionaries
        """
        mul = 0.0
        for word in dict_u:
            if word in dict_v:
                mul += dict_u[word] * dict_v[word]
        return mul

    if len(dict_x) == 0 or len(dict_y) == 0: return 0.0
    return multiply(dict_x, dict_y) / (np.sqrt(multiply(dict_x, dict_x)) * np.sqrt(multiply(dict_y, dict_y)))


def main():
    # Read Preprocessed Data
    quetions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")

    pair = list(zip(quetions, pred_questions, answers, pred_answers))
    random.shuffle(pair)
    quetions, pred_questions, answers, pred_answers = zip(*pair)

    # Split Data
    split_ratio = 0.7
    split_len = int(len(quetions) * split_ratio)
    train_questions = quetions[:split_len]
    train_pred_questions = pred_questions[:split_len]
    train_answers = answers[:split_len]
    train_pred_answers = pred_answers[:split_len]
    test_questions = quetions[split_len:]
    test_pred_questions = pred_questions[split_len:]
    test_answers = answers[split_len:]
    test_pred_answers = pred_answers[split_len:]

    # Calculate TF-IDF
    idf_dict = generate_idf_dict(train_pred_questions)
    tf_idf_train_pred_questions = generate_tf_idf_list(train_pred_questions, idf_dict)
    tf_idf_test_pred_questions = generate_tf_idf_list(test_pred_questions, idf_dict)

    # Build word --> sentence dictionary
    word_sentence_dict = generate_word_sentence_dict(train_pred_questions)
    # print(word_sentence_dict)

    # Choose the Top K similar ones
    top_k = 5
    output = open("Data/TFIDF.txt", 'w')
    for i in range(len(tf_idf_test_pred_questions)):
        top = []

        # Generate sentence id set which include at least one same word
        sentence_id_set = set()
        for j in range(len(test_pred_questions[i])):
            if test_pred_questions[i][j] in word_sentence_dict:
                sentence_id_set.update(word_sentence_dict[test_pred_questions[i][j]])
                # print test_pred_questions[i][j],
        # print(len(sentence_id_set))

        # Generate cosine similarity score
        for j in sentence_id_set:
            score = cosine_similarity(tf_idf_test_pred_questions[i], tf_idf_train_pred_questions[j])
            heapq.heappush(top, (-score, str(j)))

        output.write("Question: " + test_questions[i].encode("utf-8") + "\n")
        output.write("Ground Truth: " + test_answers[i].encode("utf-8") + "\n")

        # Generate Top K
        for j in range(min(top_k,len(top))):
            item = int(heapq.heappop(top)[1])
            output.write("Our similar " + str(j + 1) + ": " + train_questions[item].encode("utf-8") + "\n")
            output.write("Our reply " + str(j + 1) + ": " + train_answers[item].encode("utf-8") + "\n")
        output.write("\n")

    output.close()


if __name__ == "__main__":
    # main()
    tfidf = TFIDF()
    tfidf.ask_response("有什么好的电脑么")
    tfidf.ask_response("有什么推荐的手机么")
# This Python file uses the following encoding: utf-8
import csv
import numpy as np
import random


class dataset(object):
    def __init__(self, s1, s2, label):
        self.index_in_epoch = 0
        self.s1 = s1
        self.s2 = s2
        self.label = label
        self.example_nums = len(label)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.example_nums:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.example_nums)

            np.random.shuffle(perm)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_nums
        end = self.index_in_epoch
        return np.array(self.s1[start:end]), np.array(self.s2[start:end]), np.array(self.label[start:end])


def read_origin_data(input_file_name, output_file_name, stopwords_file="Data/Chinese Stop Words"):
    """
    Read file and preprocessed data
    Output preprocessed data
    """
    file = open(input_file_name)
    questions, answers = [], []

    for i, line in enumerate(file.readlines()):
        qa = line.strip().split("	")
        # No qa
        if len(qa) == 0: continue

        questions.append(qa[0].decode('utf-8'))

        # No answer
        if len(qa) == 1:
            answers.append("")
            continue

        answers.append(qa[1].decode('utf-8'))

        # Counter for test
        # if i >100: break

    file.close()
    print("Read files End")

    pred_questions = preprocessing(questions, stopwords_file)
    pred_answers = preprocessing(answers, stopwords_file)

    # Output
    with open(output_file_name, 'w') as csvfile:
        fieldnames = ['question', 'pred_question', 'answer', 'pred_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(len(questions)):
            pred_q = ""
            pred_a = ""
            for j in range(len(pred_questions[i])):
                pred_q += pred_questions[i][j] + " "
            for j in range(len(pred_answers[i])):
                pred_a += pred_answers[i][j] + " "
            writer.writerow({
                'question': questions[i].encode("utf-8"),
                'pred_question': pred_q.encode("utf-8"),
                'answer': answers[i].encode("utf-8"),
                'pred_answer': pred_a.encode("utf-8")
            })

    return questions, pred_questions, answers, pred_answers


def read_pred_data(file_name):
    """
    Read file with preprocessed data
    """
    quetions, pred_questions, answers, pred_answers = [], [], [], []
    with open(file_name, 'r') as csvfile:
        file_info = csv.reader(csvfile)
        # Store the information
        for i, line in enumerate(file_info):
            if i == 0: continue
            quetions.append(line[0].strip().decode("utf-8"))
            pred_questions.append(line[1].strip().decode("utf-8").split(" "))
            answers.append(line[2].strip().decode("utf-8"))
            pred_answers.append(line[3].strip().decode("utf-8").split(" "))

            # Counter for test
            if i > 100000: break

    return quetions, pred_questions, answers, pred_answers

def padding_sentence(s1, s2, seq_length):
    """
    Padding each sentence to the max sentence length with <unk> 0
    """

    sentence_num = len(s1)
    s1_padding = np.zeros([sentence_num, seq_length], dtype=int)
    s2_padding = np.zeros([sentence_num, seq_length], dtype=int)

    for i, s in enumerate(s1):
        min_length = min(len(s), seq_length)
        s1_padding[i][:min_length] = s[:min_length]

    for i, s in enumerate(s2):
        min_length = min(len(s), seq_length)
        s2_padding[i][:min_length] = s[:min_length]

    print("padding completed")
    return s1_padding, s2_padding

def generate_word_embedding(questions, answers, dimension):
    """
    Generate word embedding matrix based on questions and answers
    """
    word_dict = {'<unk>': 0}
    for i in range(len(questions)):
        for j in range(len(questions[i])):
            if questions[i][j] not in word_dict:
                word_dict[questions[i][j]] = len(word_dict)

    for i in range(len(answers)):
        for j in range(len(answers[i])):
            if answers[i][j] not in word_dict:
                word_dict[answers[i][j]] = len(word_dict)

    word_embedding = np.random.normal(size=(len(word_dict), dimension))

    return word_dict, word_embedding


def generate_cnn_data(questions, answers, word_dict, neg_sample_ratio, seq_length):
    """
    Generate QA pair data
    """
    s1, s2, score = [], [], []

    # positive sampling
    for i in range(len(questions)):
        q, a = [], []
        for j in range(len(questions[i])):
            if questions[i][j] in word_dict:
                q.append(word_dict[questions[i][j]])
            else:
                q.append(0)
        for j in range(len(answers[i])):
            if answers[i][j] in word_dict:
                a.append(word_dict[answers[i][j]])
            else:
                a.append(0)
        s1.append(q)
        s2.append(a)
        score.append([1])

    # negative sampling
    for i in range(len(questions) * neg_sample_ratio):
        q, a = [], []
        q_index = i % len(questions)
        a_index = int(random.random() * len(answers))
        # print(q_index, a_index)
        for j in range(len(questions[q_index])):
            if questions[q_index][j] in word_dict:
                q.append(word_dict[questions[q_index][j]])
            else:
                q.append(0)
        for j in range(len(answers[a_index])):
            if answers[a_index][j] in word_dict:
                a.append(word_dict[answers[a_index][j]])
            else:
                a.append(0)
        s1.append(q)
        s2.append(a)
        score.append([0])

    s1, s2 = padding_sentence(s1, s2, seq_length)
    print("Sampling completed")

    return s1, s2, score


def get_stop_words(file_name):
    """
    Chinese Stop words from file
    """
    file = open(file_name)
    stop_words = []
    for line in file.readlines():
        stop_words.append(line.strip())
    file.close()
    return set(stop_words)


def preprocessing(conversations, stopwords_file="Data/Chinese Stop Words"):
    """
    Stop words removal
    """
    pred_conversations = []
    stop_words = get_stop_words(stopwords_file)
    for i in range(len(conversations)):
        pred_conversation = []
        for j in range(len(conversations[i])):
            if conversations[i][j].encode('utf-8') in stop_words: continue
            if conversations[i][j] == " ": continue
            pred_conversation.append(conversations[i][j])
        pred_conversations.append(pred_conversation)
    return pred_conversations


def main():
    # read_origin_data("Data/QA-pair","Data/simple_pred_QA-pair.csv", stopwords_file="Data/Simple Chinese Stop Words.txt")
    # stop_words = get_stop_words("Data/Chinese Stop Words")
    # pred_conversations = preprocessing([u'你好？！你呢'])
    read_pred_data("Data/pred_QA-pair.csv")


if __name__ == "__main__":
    main()

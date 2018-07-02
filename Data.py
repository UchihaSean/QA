# This Python file uses the following encoding: utf-8
import csv

def read_origin_data(input_file_name,output_file_name, stopwords_file = "Data/Chinese Stop Words"):
    """
    Read file and preprocessed data
    Output preprocessed data
    """
    file = open(input_file_name)
    questions, answers = [], []

    for i,line in enumerate(file.readlines()):
        qa = line.strip().split("	")
        # No qa
        if len(qa)==0: continue

        questions.append(qa[0].decode('utf-8'))

        # No answer
        if len(qa)==1:
            answers.append("")
            continue

        answers.append(qa[1].decode('utf-8'))

        # Counter for test
        if i >100: break

    file.close()
    print("Read files End")


    pred_questions = preprocessing(questions,stopwords_file)
    pred_answers = preprocessing(answers,stopwords_file)

    # Output
    with open(output_file_name,'w') as csvfile:
        fieldnames = ['question','pred_question','answer','pred_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(len(questions)):
            pred_q = ""
            pred_a = ""
            for j in range(len(pred_questions[i])):
                pred_q+=pred_questions[i][j]+" "
            for j in range(len(pred_answers[i])):
                pred_a+=pred_answers[i][j]+" "
            writer.writerow({
                'question':questions[i].encode("utf-8"),
                'pred_question':pred_q.encode("utf-8"),
                'answer':answers[i].encode("utf-8"),
                'pred_answer':pred_a.encode("utf-8")
            })


    return questions, pred_questions, answers, pred_answers

def read_pred_data(file_name):
    """
    Read file with preprocessed data
    """
    quetions, pred_questions, answers, pred_answers =[], [], [], []
    with open(file_name, 'r') as csvfile:
        file_info = csv.reader(csvfile)
        # Store the information
        for i,line in enumerate(file_info):
            if i == 0: continue
            quetions.append(line[0].strip().decode("utf-8"))
            pred_questions.append(line[1].strip().decode("utf-8").split(" "))
            answers.append(line[2].strip().decode("utf-8"))
            pred_answers.append(line[3].strip().decode("utf-8").split(" "))

            # Counter for test
            if i>1000: break

    return quetions, pred_questions, answers, pred_answers


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





def preprocessing(conversations, stopwords_file):
    """
    Stop words removal
    """
    pred_conversations = []
    stop_words = get_stop_words(stopwords_file)
    for i in range(len(conversations)):
        pred_conversation = []
        for j in range(len(conversations[i])):
            if conversations[i][j].encode('utf-8') in stop_words: continue
            if conversations[i][j] ==" ": continue
            pred_conversation.append(conversations[i][j])
        pred_conversations.append(pred_conversation)
    return pred_conversations






def main():
    read_origin_data("Data/QA-pair","Data/simple_pred_QA-pair.csv", stopwords_file="Data/Simple Chinese Stop Words.txt")
    # stop_words = get_stop_words("Data/Chinese Stop Words")
    # pred_conversations = preprocessing([u'你好？！你呢'])
    # read_pred_data("Data/pred_QA-pair.csv")


if __name__ == "__main__":
    main()

# coding=utf-8
# ! /usr/bin/env python

import datetime
import os
import time
import numpy as np
import tensorflow as tf
import heapq

import Data
from CNN_model import TextCNN
import random

class CNN:

    def __init__(self, top_k = 3, questions = None, pred_questions = None, answers = None, pred_answers = None):

        # Parameters
        self.train_sample_percentage = 0.8
        self.dev_sample_percentage = 0.1
        self.data_file = "Data/simple_pred_QA-pair.csv"
        self.filter_sizes = "3,4,5"
        self.num_filters = 128
        self.seq_length = 32
        self.num_classes = 1
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 1
        self.batch_size = 64
        self.num_epochs = 200
        self.evaluate_every = 100
        self.checkpoint_every = 100
        self.num_checkpoints = 5
        self.allow_soft_placement = True
        self.log_device_placement = False
        self.embedding_dimension = 300
        self.neg_sample_ratio = 5
        self.epoch_num = 10000
        self.questions = questions
        self.pred_questions = pred_questions
        self.answers = answers
        self.pred_answers = pred_answers
        self.top_k = top_k

        # Random seed
        random.seed(12345)

        # Data
        self.data_preparation()

    def data_preparation(self):
        """
        Read Data and split
        """

        # Read Preprocessed Data
        print("Loading data...")
        if self.questions == None:
            self.questions, self.pred_questions, self.answers, self.pred_answers = Data.read_pred_data(self.data_file)

        # self.word_dict, self.word_embedding = Data.generate_word_embedding(self.pred_questions, self.pred_answers, self.embedding_dimension)

        # Get word embeding
        self.word_dict, self.word_embedding = Data.read_single_word_embedding("Data/single_word_embedding")

        # Generate Data for CNN
        self.s1, self.s2, self.score = Data.generate_cnn_data(self.pred_questions, self.pred_answers, self.word_dict, self.neg_sample_ratio, self.seq_length)

        # Shuffle data with seed
        pair = list(zip(self.s1, self.s2, self.score))
        random.shuffle(pair)
        self.s1, self.s2, self.score = zip(*pair)
        self.s1, self.s2, self.score = np.array(self.s1), np.array(self.s2), np.array(self.score)

        sample_num = len(self.score)
        train_end = int(sample_num * self.train_sample_percentage)
        dev_end = int(sample_num * (self.train_sample_percentage+self.dev_sample_percentage))

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        self.s1_train, self.s1_dev, self.s1_test = self.s1[:train_end], self.s1[train_end:dev_end], self.s1[dev_end:]
        self.s2_train, self.s2_dev, self.s2_test = self.s2[:train_end], self.s2[train_end:dev_end], self.s2[dev_end:]
        self.score_train, self.score_dev, self.score_test = self.score[:train_end], self.score[train_end:dev_end], self.score[dev_end:]
        print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(self.score_train), len(self.score_dev), len(self.score_test)))

        # Build word --> sentence dictionary
        self.word_sentence_dict = Data.generate_word_sentence_dict(self.pred_answers)


    def train_dev(self):
        """
        Train CNN model and Score on the dev
        """
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=self.seq_length,
                    num_classes=self.num_classes,
                    filter_sizes=list(map(int, self.filter_sizes.split(","))),
                    num_filters=self.num_filters,
                    word_embedding= self.word_embedding,
                    l2_reg_lambda=self.l2_reg_lambda)


                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


                saver = tf.train.Saver()

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Restore
                # saver.restore(sess, "/tmp/model/ckpt")
                # print("Restore model information")

                # Embedding output
                # output = open("Data/word_embedding_before.txt", 'w')
                # output.write(sess.run(cnn.W))
                # output.close()



                def train_step(s1, s2, score):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_s1: s1,
                        cnn.input_s2: s2,
                        cnn.input_y: score,
                        cnn.dropout_keep_prob: self.dropout_keep_prob
                    }
                    _, step, loss, pearson = sess.run(
                        [train_op, global_step, cnn.loss, cnn.pearson],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, pearson {:g}".format(time_str, step, loss, pearson))
                    # train_summary_writer.add_summary(summaries, step)


                def dev_step(s1, s2, score):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_s1: s1,
                        cnn.input_s2: s2,
                        cnn.input_y: score,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, loss, pearson = sess.run(
                        [global_step, cnn.loss, cnn.pearson],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, pearson {:g}".format(time_str, step, loss, pearson))



                # Generate batches
                STS_train = Data.dataset(s1=self.s1_train, s2=self.s2_train, label=self.score_train)
                # Training loop. For each batch...

                for i in range(self.epoch_num):
                    batch_train = STS_train.next_batch(self.batch_size)

                    train_step(batch_train[0], batch_train[1], batch_train[2])
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(self.s1_dev, self.s2_dev, self.score_dev)
                        print("")

                save_path = saver.save(sess, "/tmp/model/ckpt")
                print("Save model to "+save_path)

                # Embedding output
                # output = open("Data/word_embedding_after.txt", 'w')
                # output.write(sess.run(cnn.W))
                # output.close()
    def test(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=self.seq_length,
                    num_classes=self.num_classes,
                    filter_sizes=list(map(int, self.filter_sizes.split(","))),
                    num_filters=self.num_filters,
                    word_embedding= self.word_embedding,
                    l2_reg_lambda=self.l2_reg_lambda)


                saver = tf.train.Saver()

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Restore
                saver.restore(sess, "/tmp/model/ckpt")
                print("Restore model information")

                def test_step(s1, s2, score):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_s1: s1,
                        cnn.input_s2: s2,
                        cnn.input_y: score,
                        cnn.dropout_keep_prob: 1.0
                    }
                    loss, pearson = sess.run(
                        [cnn.real_loss, cnn.pearson],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: loss {:g}, pearson {:g}".format(time_str, loss, pearson))

                print("\nTest")
                test_step(self.s1_test, self.s2_test, self.score_test)


    def ask_response(self, question):
        """
        :param question: input a question
        :return: top k response
         """
        def get_score(s1, s2):
            """
            Get CNN similarity score based on two sentences
            """
            with tf.Graph().as_default():
                sess = tf.Session()
                with sess.as_default():
                    cnn = TextCNN(
                        sequence_length=self.seq_length,
                        num_classes=self.num_classes,
                        filter_sizes=list(map(int, self.filter_sizes.split(","))),
                        num_filters=self.num_filters,
                        l2_reg_lambda=self.l2_reg_lambda)

                    saver = tf.train.Saver()
                    # Restore
                    saver.restore(sess, "/tmp/model/ckpt")
                    feed_dict = {
                        cnn.input_s1: s1,
                        cnn.input_s2: s2,
                        cnn.dropout_keep_prob: 1.0
                    }
                    scores = sess.run(cnn.scores,feed_dict)
                return scores[0]

        top = []
        pred_q = Data.preprocessing([question.decode("utf-8")])

        # Generate sentence id set which include at least one same word
        sentence_id_set = set()
        for j in range(len(pred_q[0])):
            if pred_q[0][j] in self.word_sentence_dict:
                sentence_id_set.update(self.word_sentence_dict[pred_q[0][j]])

        print(len(sentence_id_set))
        for i in sentence_id_set:
            s1, s2 = Data.generate_cnn_sentence(question.decode("utf-8"), self.answers[i], self.word_dict,self.seq_length)
            score = get_score(s1, s2)
            # print(score)
            heapq.heappush(top, (-score, str(i)))

        # print("Question: %s" % question)

        response = []
        # Generate Top K
        for j in range(min(self.top_k, len(top))):
            item = int(heapq.heappop(top)[1])
            # print("Similar %d: %s" % (j + 1, self.questions[item]))
            # print("CNN Response %d: %s" % (j + 1, self.answers[item]))
            response.append(self.answers[item])

        # print("")

        return response




def main():
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("Data/pred_QA-pair.csv")
    cnn = CNN(3, questions, pred_questions, answers, pred_answers)
    cnn.train_dev()
    # cnn.test()
    # cnn.ask_response("有什么好的电脑么")


if __name__ == "__main__":
    main()
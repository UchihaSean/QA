# coding=utf-8
# ! /usr/bin/env python

import datetime
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import Data
from CNN_model import TextCNN
import random

class CNN:

    def __init__(self):
        # Parameters
        # ==================================================
        self.train_sample_percentage = 0.9
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
        self.embedding_dimension = 50
        self.neg_sample_ratio = 5
        self.epoch_num = 3000

    def data_preparation(self):
        # Read Preprocessed Data
        print("Loading data...")
        self.quetions, self.pred_questions, self.answers, self.pred_answers = Data.read_pred_data(self.data_file)

        self.word_dict, self.word_embedding = Data.generate_word_embedding(self.pred_questions, self.pred_answers, self.embedding_dimension)

        self.s1, self.s2, self.score = Data.generate_cnn_data(self.pred_questions, self.pred_answers, self.word_dict, self.neg_sample_ratio, self.seq_length)


        pair = list(zip(self.s1, self.s2, self.score))
        random.shuffle(pair)
        self.s1, self.s2, self.score = zip(*pair)
        self.s1, self.s2, self.score = np.array(self.s1), np.array(self.s2), np.array(self.score)

        sample_num = len(self.score)
        train_end = int(sample_num * self.train_sample_percentage)

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        self.s1_train, self.s1_dev = self.s1[:train_end], self.s1[train_end:]
        self.s2_train, self.s2_dev = self.s2[:train_end], self.s2[train_end:]
        self.score_train, self.score_dev = self.score[:train_end], self.score[train_end:]
        print("Train/Dev split: {:d}/{:d}".format(len(self.score_train), len(self.score_dev)))


    def train_dev(self):
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
                    l2_reg_lambda=self.l2_reg_lambda)

                cnn.set_word_embedding(self.word_embedding)
                cnn.initial()

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
                saver.restore(sess, "/tmp/model/ckpt")
                print("Restore model information")




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
                    # if writer:
                    #     writer.add_summary(summaries, step)


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

    def ask_response(self, question):

        def get_score(s1, s2):
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
                    print("Restore model information")
                    feed_dict = {
                        cnn.input_s1: s1,
                        cnn.input_s2: s2,
                        cnn.dropout_keep_prob: 1.0
                    }
                    scores = sess.run(cnn.scores,feed_dict)
                return scores[0]




def main():
    cnn = CNN()
    cnn.data_preparation()
    cnn.train_dev()


if __name__ == "__main__":
    main()
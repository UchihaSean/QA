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

# Parameters
# ==================================================

train_sample_percentage = 0.9
data_file = "Data/simple_pred_QA-pair.csv"
filter_sizes = "3,4,5"
num_filters = 128
seq_length = 36
num_classes = 1
dropout_keep_prob = 0.5
l2_reg_lambda = 1
batch_size = 64
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
allow_soft_placement = True
log_device_placement = False
embedding_dimension = 50

# Data Preparation
# ==================================================

# Read Preprocessed Data
print("Loading data...")
quetions, pred_questions, answers, pred_answers = Data.read_pred_data(data_file)

# pair = list(zip(quetions, pred_questions, answers, pred_answers))
# random.shuffle(pair)
# quetions, pred_questions, answers, pred_answers = zip(*pair)

word_dict, word_embedding = Data.generate_word_embedding(pred_questions, pred_answers, embedding_dimension)

s1, s2, score = Data.generate_cnn_data(pred_questions, pred_answers, word_dict)
sample_num = len(score)
train_end = int(sample_num * train_sample_percentage)

# Split train/test set
# TODO: This is very crude, should use cross-validation
s1_train, s1_dev = s1[:train_end], s1[train_end:]
s2_train, s2_dev = s2[:train_end], s2[train_end:]
score_train, score_dev = score[:train_end], score[train_end:]
print("Train/Dev split: {:d}/{:d}".format(len(score_train), len(score_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=seq_length,
            num_classes=num_classes,
            filter_sizes=list(map(int, filter_sizes.split(","))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and pearson
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("pearson", cnn.pearson)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        def train_step(s1, s2, score):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_s1: s1,
                cnn.input_s2: s2,
                cnn.input_y: score,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, pearson = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.pearson],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, pearson {:g}".format(time_str, step, loss, pearson))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(s1, s2, score, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_s1: s1,
                cnn.input_s2: s2,
                cnn.input_y: score,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, pearson = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.pearson],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, pearson))
            if writer:
                writer.add_summary(summaries, step)


        # Generate batches
        STS_train = CNN_data_helper.dataset(s1=s1_train, s2=s2_train, label=score_train)
        # Training loop. For each batch...

        for i in range(40000):
            batch_train = STS_train.next_batch(batch_size)

            train_step(batch_train[0], batch_train[1], batch_train[2])
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(s1_dev, s2_dev, score_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

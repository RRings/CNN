#-*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time
import datetime
from tensorflow import flags
import tensorflow as tf
import numpy as np
import cnn_tool as tool

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 단어 수
        - embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        """
        <Variable>
            - W: 각 단어의 임베디드 벡터의 성분을 랜덤하게 할당
        """
        #with tf.device('/gpu:0'), tf.name_scope("embedding"):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")    # xw_plus_b = matmul(x, W) + b
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# data loading
data_path = '~/Dev/tensorflow/inflearn/RRings/ratings_test_22.csv'
contents, points = tool.loading_rdata(data_path)
contents = tool.cut(contents,cut=2)


# tranform document to vector
max_document_length = 200
x, vocabulary, vocab_size = tool.make_input(contents,max_document_length)

#print(vocab_size)
#print(vocabulary)

#print('사전단어수 : %s' % (vocab_size))
y = tool.make_output(points,threshold=0.5)   # 긍정 부정으로 0, 1로만 구분되어 있어서 0.5 를 선택함

# divide dataset into train/test set
#x_train, x_test, y_train, y_test = tool.divide(x,y,train_prop=0.8) # 80% 는 train   20% 는 test


x_test, y_test = tool.divide(x,y)

# Model Hyperparameters
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedded vector (default: 128)")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
'''
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
'''
# 3. train the model and test
with tf.Graph().as_default():
    sess = tf.Session()

    with sess.as_default():

        cnn = TextCNN(sequence_length=200,
                      num_classes=2,
                      vocab_size=vocab_size,
                      embedding_size=FLAGS.embedding_dim,
                      filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())



        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }

            loss, accuracy = sess.run([cnn.loss, cnn.accuracy],feed_dict)
          #  print(sess.run([cnn.loss, cnn.accuracy],feed_dict))
            a = sess.run([cnn.accuracy], feed_dict)
            if a == [1.0] :
                print('이 문장은 긍정적이네요 !')
            else:
                print('이 문장은 부정적이네요 ')
            #print(sess.run([cnn.accuracy], feed_dict))
            time_str = datetime.datetime.now().isoformat()
           # print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))

        modelName = "mnist.pd"
        saver.restore(sess, modelName)
        dev_step(x_test, y_test)





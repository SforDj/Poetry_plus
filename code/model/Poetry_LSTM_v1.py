import tensorflow as tf
import numpy as np
from code.data_handler.data_prepare import Poetry
import datetime
import os
import random

class Poetry_LSTM_v1:
    def __init__(self, config):
        self.poetry_file = config.poetry_file
        self.batch_size = config.batch_size
        self.n_rnn_cell = config.n_rnn_cell
        self.keep_prob = config.keep_prob
        self.poetry = Poetry(self.poetry_file, self.batch_size)
        self.word_len = len(self.poetry.word_to_int)

    @staticmethod
    def transfer_to_word_embeddings(inputs, n_rnn_cell, word_len):
        with tf.variable_scope('word_embedding'):
            with tf.device("/cpu:0"):
                embeddings = tf.get_variable('embedding', [word_len, n_rnn_cell])
                rnn_inputs = tf.nn.embedding_lookup(embeddings, inputs)
        return rnn_inputs

    @staticmethod
    def softmax_variable(n_rnn_cell, word_len):
        with tf.variable_scope('softmax_variable'):
            w = tf.get_variable("w", [n_rnn_cell, word_len])
            b = tf.get_variable("b", [word_len])
        return w, b

    def lstm_graph(self, rnn_inputs, keep_prob):
        single_layer_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_rnn_cell)
        dropped_lstm = tf.nn.rnn_cell.DropoutWrapper(single_layer_lstm, output_keep_prob=keep_prob)

        lstm = tf.nn.rnn_cell.MultiRNNCell([dropped_lstm] * 2)
        initial_states = lstm.zero_state(self.batch_size, tf.float32)
        output, state = tf.nn.dynamic_rnn(lstm, rnn_inputs, initial_state=initial_states)

        seq_output = tf.concat(output, 1)
        x = tf.reshape(seq_output, [-1, self.n_rnn_cell])

        w, b = self.softmax_variable(self.n_rnn_cell, self.word_len)
        logits = tf.matmul(x, w) + b
        preds = tf.nn.softmax(logits, name='predictions')
        return state, logits, preds

    def loss_graph(self, targets, logits):
        y_one_hot = tf.one_hot(targets, self.word_len)
        y_reshaped = tf.reshape(y_one_hot, [-1, self.word_len])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        return loss

    def optimizer_graph(self, loss, learning_rate):
        grad_clip = 5
        all_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, all_variables), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, all_variables))
        return optimizer

    def train(self, epoch):
        inputs = tf.placeholder(tf.int32, (self.batch_size, None))
        targets = tf.placeholder(tf.int32, (self.batch_size, None))

        lstm_inputs = self.transfer_to_word_embeddings(inputs, self.n_rnn_cell, self.word_len)

        state, logits, preds = self.lstm_graph(self.n_rnn_cell, self.keep_prob)
        loss = self.loss_graph(targets, logits)
        learning_rate = tf.Variable(0.0, trainable=False)
        optimizer = self.optimizer_graph(loss, learning_rate)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables)
            step = 0
            for i in range(epoch):
                batches = self.poetry.get_batch()
                sess.run(tf.assign(learning_rate, 0.001 * (0.97 ** i)))
                for batch_x, batch_y in batches:
                    feed = {inputs: batch_x, targets: batch_y}
                    batch_loss, _, new_state = sess.run([loss, optimizer, state], feed_dict=feed)
                    print(datetime.datetime.now().strftime('%c'), ' i:', i, 'step:', step, ' batch_loss:', batch_loss)
                    step += 1
            model_path = os.getcwd() + os.sep + "poetry.model"
            saver.save(sess, model_path, global_step=step)
            sess.close()

    def gen(self, poem_len):
        def to_word(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1) * s))
            return self.poetry.int_to_word[sample]

        self.batch_size = 1
        inputs = tf.placeholder(tf.int32, shape=(self.batch_size, 1), name='inputs')
        self.keep_prob = 1.0
        lstm_inputs = self.transfer_to_word_embeddings(inputs, self.n_rnn_cell, self.word_len)
        state, logits, preds = self.lstm_graph(self.n_rnn_cell, self.keep_prob)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))

            x = np.zeros((1, 1))
            x[0, 0] = self.poetry.word_to_int[self.poetry.int_to_word[random.randint(1, self.word_len - 1)]]
            feed = {inputs: x}

            new_pred, new_state = sess.run([preds, state], feed_dict=feed)
            word = to_word(new_pred)
            poem = self.poetry.int_to_word[x[0, 0]]
            while len(poem) < poem_len:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = self.poetry.word_to_int[word]
                feed = {inputs: x}
                new_pred, new_state = sess.run([preds, state], feed_dict=feed)
                word = to_word(new_pred)
            return poem







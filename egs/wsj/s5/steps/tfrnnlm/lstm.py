# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright (C) 2017 Intellisist, Inc. (Author: Hainan Xu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# this script trains a vanilla RNNLM with TensorFlow. 
# to call the script, do
# python steps/tfrnnlm/lstm.py --data-path=$datadir \
#        --save-path=$savepath --vocab-path=$rnn.wordlist [--hidden-size=$size]
#
# One example recipe is at egs/ami/s5/local/tfrnnlm/run_lstm.sh

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import inspect
import time

import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("hidden-size", 200, "hidden dim of RNN")

flags.DEFINE_string("data-path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("vocab-path", None,
                    "Where the wordlist file is stored.")
flags.DEFINE_string("save-path", None,
                    "Model output directory.")
flags.DEFINE_bool("use-fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

class Config(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 64

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class RnnlmInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.rnnlm_producer(
        data, batch_size, num_steps, name=name)


class RnnlmModel(object):
  """The RNNLM model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    self.cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = self.cell.zero_state(batch_size, data_type())
    self._initial_state_single = self.cell.zero_state(1, data_type())

    self.initial = tf.reshape(tf.stack(axis=0, values=self._initial_state_single), [config.num_layers, 2, 1, size], name="test_initial_state")


    # first implement the less efficient version
    test_word_in = tf.placeholder(tf.int32, [1, 1], name="test_word_in")

    state_placeholder = tf.placeholder(tf.float32, [config.num_layers, 2, 1, size], name="test_state_in")
    # unpacking the input state context 
    l = tf.unstack(state_placeholder, axis=0)
    test_input_state = tuple(
               [tf.contrib.rnn.LSTMStateTuple(l[idx][0],l[idx][1])
                 for idx in range(config.num_layers)]
    )

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())

      inputs = tf.nn.embedding_lookup(self.embedding, input_.input_data)
      test_inputs = tf.nn.embedding_lookup(self.embedding, test_word_in)

    # test time
    with tf.variable_scope("RNN"):
      (test_cell_output, test_output_state) = self.cell(test_inputs[:, 0, :], test_input_state)

    test_state_out = tf.reshape(tf.stack(axis=0, values=test_output_state), [config.num_layers, 2, 1, size], name="test_state_out")
    test_cell_out = tf.reshape(test_cell_output, [1, size], name="test_cell_out")
    # above is the first part of the graph for test
    # test-word-in
    #               > ---- > test-state-out
    # test-state-in        > test-cell-out


    # below is the 2nd part of the graph for test
    # test-word-out
    #               > prob(word | test-word-out)
    # test-cell-in

    test_word_out = tf.placeholder(tf.int32, [1, 1], name="test_word_out")
    cellout_placeholder = tf.placeholder(tf.float32, [1, size], name="test_cell_in")

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())

    test_logits = tf.matmul(cellout_placeholder, softmax_w) + softmax_b
    test_softmaxed = tf.nn.log_softmax(test_logits)

    p_word = test_softmaxed[0, test_word_out[0,0]]
    test_out = tf.identity(p_word, name="test_out")

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(
    #     cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > -1: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = self.cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  return Config()

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to RNNLM data directory")

  raw_data = reader.rnnlm_raw_data(FLAGS.data_path, FLAGS.vocab_path)
  train_data, valid_data, _, word_map = raw_data

  config = get_config()
  config.hidden_size = FLAGS.hidden_size
  config.vocab_size = len(word_map)
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = RnnlmInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = RnnlmModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = RnnlmInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = RnnlmModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)

        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path)

if __name__ == "__main__":
  tf.app.run()

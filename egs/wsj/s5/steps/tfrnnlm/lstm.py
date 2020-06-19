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
# python steps/tfrnnlm/lstm.py --data_path=$datadir \
#        --save_path=$savepath --vocab_path=$rnn.wordlist [--hidden-size=$size]
#
# One example recipe is at egs/ami/s5/local/tfrnnlm/run_lstm.sh

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import absl.flags as flags
import tensorflow as tf

import reader

flags.DEFINE_integer("hidden_size", 200, "hidden dim of RNN")

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("vocab_path", None,
                    "Where the wordlist file is stored.")
flags.DEFINE_string("save_path", "export",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
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


class RNNLMModel(tf.Module):
  """The RNN model itself."""

  def __init__(self, config, logits_bias_initializer=None):
    super().__init__()
    self._config = config

    size = config.hidden_size
    vocab_size = config.vocab_size
    dt = data_type()

    def lstm_cell():
      return tf.keras.layers.LSTMCell(size, dtype=dt, unit_forget_bias=False)

    def add_dropout(cell):
      if config.keep_prob < 1:
        cell = tf.nn.RNNCellDropoutWrapper(cell=cell, output_keep_prob=config.keep_prob)
      return cell

    self.embedding = tf.keras.layers.Embedding(vocab_size, size, dtype=dt)
    self.cells = [lstm_cell() for _ in range(config.num_layers)]
    self.rnn = tf.keras.layers.RNN(self.cells, return_sequences=True)

    if logits_bias_initializer is None:
      logits_bias_initializer = 'zeros'
    self.fc = tf.keras.layers.Dense(vocab_size, bias_initializer=logits_bias_initializer)

    # only used in training
    self.training_cells = [add_dropout(cell) for cell in self.cells]
    self.training_rnn = tf.keras.layers.RNN(self.training_cells, return_sequences=True)

  def get_logits(self, word_ids, is_training=False):
    rnn = self.training_rnn if is_training else self.rnn
    inputs = self.embedding(word_ids)
    if is_training and self._config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, 1 - self._config.keep_prob)
    rnn_out = rnn(inputs)
    logits = self.fc(rnn_out)
    return logits

  def get_loss(self, word_ids, labels, is_training=False):
    logits = self.get_logits(word_ids, is_training)
    loss_obj = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_obj(labels, logits)

  def get_score(self, logits):
    """Take logits as input, output a score."""
    return tf.nn.log_softmax(logits)

  @tf.function
  def get_initial_state(self):
    """Exported function which emits zeroed RNN context vector."""
    # This seems a bug in TensorFlow, but passing tf.int32 makes the state tensor also int32.
    fake_input = tf.constant(0, dtype=tf.float32, shape=[1, 1])
    initial_state = tf.stack(self.rnn.get_initial_state(fake_input))
    return {"initial_state": initial_state}

  @tf.function
  def single_step(self, context, word_id):
    """Exported function which perform one step of the RNN model."""
    rnn = tf.keras.layers.RNN(self.cells, return_state=True)
    context = tf.unstack(context)
    context = [tf.unstack(c) for c in context]

    inputs = self.embedding(word_id)
    rnn_out_and_states = rnn(inputs, initial_state=context)

    rnn_out = rnn_out_and_states[0]
    rnn_states = tf.stack(rnn_out_and_states[1:])

    logits = self.fc(rnn_out)
    output = self.get_score(logits)
    log_prob = output[0, word_id[0, 0]]
    return {"log_prob": log_prob, "rnn_states": rnn_states, "rnn_out": rnn_out}


class RNNLMModelTrainer(tf.Module):
  """This class contains training code."""

  def __init__(self, model: RNNLMModel, config):
    super().__init__()
    self.model = model
    self.learning_rate = tf.Variable(1e-3, dtype=tf.float32, trainable=False)
    self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
    self.max_grad_norm = config.max_grad_norm

    self.eval_mean_loss = tf.metrics.Mean()

  def train_one_epoch(self, data_producer, learning_rate, verbose=True):
    print("start epoch with learning rate {}".format(learning_rate))
    self.learning_rate.assign(learning_rate)

    for i, (inputs, labels) in enumerate(data_producer.iterate()):
      loss = self._train_step(inputs, labels)
      if verbose and i % (data_producer.epoch_size // 10) == 1:
        print("{}/{}: loss={}".format(i, data_producer.epoch_size, loss))

  @tf.function
  def evaluate(self, data_producer):
    self.eval_mean_loss.reset_states()
    for i, (inputs, labels) in enumerate(data_producer.iterate()):
      loss = self.model.get_loss(inputs, labels)
      self.eval_mean_loss.update_state(loss)

    return self.eval_mean_loss.result()

  @tf.function
  def _train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      loss = self.model.get_loss(inputs, labels, is_training=True)

    tvars = self.model.trainable_variables
    grads = tape.gradient(loss, tvars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
    self.optimizer.apply_gradients(zip(clipped_grads, tvars))
    return loss


def get_config():
  return Config()


def main(_):
  # Turn this on to try the model code with this source file itself!
  __TESTING = False

  if __TESTING:
    (train_data, valid_data), word_map = reader.rnnlm_gen_data(__file__, reader.__file__)
  else:
    if not FLAGS.data_path:
      raise ValueError("Must set --data_path to RNNLM data directory")

    raw_data = reader.rnnlm_raw_data(FLAGS.data_path, FLAGS.vocab_path)
    train_data, valid_data, _, word_map = raw_data

  config = get_config()
  config.hidden_size = FLAGS.hidden_size
  config.vocab_size = len(word_map)

  if __TESTING:
    # use a much smaller scale on our tiny test data
    config.num_steps = 8
    config.batch_size = 4

  model = RNNLMModel(config)
  train_producer = reader.RNNLMProducer(train_data, config.batch_size, config.num_steps)
  trainer = RNNLMModelTrainer(model, config)

  valid_producer = reader.RNNLMProducer(valid_data, config.batch_size, config.num_steps)

  # Save variables to disk if you want to prevent crash...
  # Data producer can also be saved to preverse feeding progress.
  checkpoint = tf.train.Checkpoint(trainer=trainer, data_feeder=train_producer)
  manager = tf.train.CheckpointManager(checkpoint, "checkpoints/", 5)

  for i in range(config.max_max_epoch):
    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
    lr = config.learning_rate * lr_decay
    trainer.train_one_epoch(train_producer, lr)
    manager.save()

    eval_loss = trainer.evaluate(valid_producer)
    print("validating: loss={}".format(eval_loss))

  # Export
  print("Saving model to %s." % FLAGS.save_path)
  spec = [tf.TensorSpec(shape=[config.num_layers, 2, 1, config.hidden_size], dtype=data_type(), name="context"),
          tf.TensorSpec(shape=[1, 1], dtype=tf.int32, name="word_id")]
  cfunc = model.single_step.get_concrete_function(*spec)
  cfunc2 = model.get_initial_state.get_concrete_function()
  tf.saved_model.save(model, FLAGS.save_path, signatures={"single_step": cfunc, "get_initial_state": cfunc2})


if __name__ == "__main__":
  absl.app.run(main)

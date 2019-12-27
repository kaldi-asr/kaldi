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
# python steps/tfrnnlm/lstm_fast.py --data_path=$datadir \
#        --save_path=$savepath --vocab_path=$rnn.wordlist [--hidden-size=$size]
#
# One example recipe is at egs/ami/s5/local/tfrnnlm/run_vanilla_rnnlm.sh

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import absl.flags as flags
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper

import reader
from lstm import RNNLMModel, RNNLMModelTrainer

# flags.DEFINE_integer("hidden_size", 200, "hidden dim of RNN")
#
# flags.DEFINE_string("data_path", None,
#                     "Where the training/test data is stored.")
# flags.DEFINE_string("vocab_path", None,
#                     "Where the wordlist file is stored.")
# flags.DEFINE_string("save_path", "export",
#                     "Model output directory.")
# flags.DEFINE_bool("use_fp16", False,
#                   "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


class Config(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.8
  batch_size = 64


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


# this new "softmax" function we show can train a "self-normalized" RNNLM where
# the sum of the output is automatically (close to) 1.0
# which saves a lot of computation for lattice-rescoring
def new_softmax(labels, logits):
  flatten_labels = tf.reshape(labels, [-1])
  n_samples = tf.shape(flatten_labels)[0]
  flatten_logits = tf.reshape(logits, shape=[n_samples, -1])
  f_logits = tf.exp(flatten_logits)
  row_sums = tf.reduce_sum(f_logits, -1) # this is the negative part of the objf

  t2 = tf.expand_dims(flatten_labels, 1)
  range = tf.expand_dims(tf.range(n_samples), 1)
  ind = tf.concat([range, t2], 1)
  res = tf.gather_nd(flatten_logits, ind)

  return -res + row_sums - 1


class MyFastLossFunction(LossFunctionWrapper):
  def __init__(self):
    super().__init__(new_softmax)


class FastRNNLMModel(RNNLMModel):
  def __init__(self, config):
    super().__init__(config, tf.constant_initializer(-9))

  def get_loss(self, word_ids, labels, is_training=False):
    logits = self.get_logits(word_ids, is_training)
    loss_obj = MyFastLossFunction()
    return loss_obj(labels, logits)

  def get_score(self, logits):
    # In this implementation, logits can be used as dist output
    return logits


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

  model = FastRNNLMModel(config)
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

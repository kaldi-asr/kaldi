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


"""Utilities for parsing RNNLM text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").split()

def _build_vocab(filename):
  words = _read_words(filename)
  word_to_id = dict(list(zip(words, list(range(len(words))))))
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def rnnlm_raw_data(data_path, vocab_path):
  """Load RNNLM raw data from data directory "data_path".

  Args:
    data_path: string path to the directory where train/valid files are stored

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to RNNLMIterator.
  """

  train_path = os.path.join(data_path, "train")
  valid_path = os.path.join(data_path, "valid")

  word_to_id = _build_vocab(vocab_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, vocabulary, word_to_id


def rnnlm_gen_data(*files):
  """Generates data and vocab from files.

  This function is used solely for testing.
  """
  import collections
  import re

  all_words = collections.Counter()
  all_word_lists = []
  for f in files:
    with open(f, mode="r") as fp:
      text = fp.read()

    word_list = re.split("[^A-Za-z]", text)
    word_list = list(filter(None, word_list))
    all_words.update(word_list)
    all_word_lists.append(word_list)

  word_to_id = {word: i for i, (word, _) in enumerate(all_words.most_common())}

  def convert(word_list):
    return [word_to_id[word] for word in word_list]

  all_word_ids = [convert(word_list) for word_list in all_word_lists]
  return all_word_ids, word_to_id


class RNNLMProducer(tf.Module):
  """This is the data feeder."""

  def __init__(self, raw_data, batch_size, num_steps, name=None):
    super().__init__(name)
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.epoch_size = (len(raw_data) - 1) // num_steps // batch_size

    # load data into a variable so that it will be separated from graph
    self._raw_data = tf.Variable(raw_data, dtype=tf.int32, trainable=False)

    ds_x = tf.data.Dataset.from_tensor_slices(self._raw_data)
    ds_y = ds_x.skip(1)
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    # form samples
    ds = ds.batch(num_steps, drop_remainder=True)
    # form batches
    self._ds = ds.batch(batch_size, drop_remainder=True)

  def iterate(self):
    return self._ds


if __name__ == "__main__":
  samples = list(range(100))
  ds = RNNLMProducer(samples, 4, 8)
  print(ds.epoch_size)
  for data in ds.iterate():
    print(data)

"""Tests for scoring module."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import os

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises_regex)

from scorelib.rttm import load_rttm
from scorelib.score import flatten_labels, score, turns_to_frames, Scores
from scorelib.turn import Turn
from scorelib.uem import UEM


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_turns_to_frames():
    # Check validation.
    turns = [
        Turn(0, 11, speaker_id='S1', file_id='FILE1'),
        Turn(5, 11, speaker_id='S2', file_id='FILE2'),
        ]
    with assert_raises_regex(ValueError, 'Turns should be from'):
        labels = turns_to_frames(turns, [(0, 1)], step=0.1)

    # Check that file containing no speech returns an (n_frames, 0) array.
    labels = turns_to_frames([], [(0, 1)], step=0.1)
    assert_equal(labels, np.zeros((10, 0), dtype='int64'))

    # Check on toy input.
    expected_labels = np.zeros((120, 2), dtype='int64')
    expected_labels[:110, 0] = 1
    expected_labels[50:110, 1] = 1
    turns = [
        Turn(0, 11, speaker_id='S1', file_id='FILE1'),
        Turn(5, 11, speaker_id='S2', file_id='FILE1'),
        ]
    labels = turns_to_frames(turns, [(0, 12)], step=0.1)
    assert_equal(labels, expected_labels)


def test_flatten_labels():
    # No speech.
    assert_equal(flatten_labels(np.zeros((5, 0), dtype='int64')),
                 np.zeros(5, dtype='int64'))
    assert_equal(flatten_labels(np.zeros((5, 1), dtype='int64')),
                 np.zeros(5, dtype='int64'))
    assert_equal(flatten_labels(np.zeros((5, 2), dtype='int64')),
                 np.zeros(5, dtype='int64'))

    # Toy input with 2 speakers.
    labels = np.array(
        [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1]],
        dtype='int64')
    assert_equal(flatten_labels(labels),
                 np.arange(4, dtype='int64'))


def test_score():
    # Some real data.
    expected_scores = Scores(
        'FILE1', 26.39309, 33.24631, 0.71880, 0.72958, 0.72415, 0.60075, 0.58534,
        0.80471, 0.72543, 0.96810, 0.55872)
    ref_turns, _, _ = load_rttm(os.path.join(TEST_DIR, 'ref.rttm'))
    sys_turns, _, _ = load_rttm(os.path.join(TEST_DIR, 'sys.rttm'))
    uem = UEM({'FILE1' : [(0, 43)]})
    file_scores, global_scores = score(ref_turns, sys_turns, uem)
    assert len(file_scores) == 1
    assert file_scores[-1].file_id == expected_scores.file_id
    assert_almost_equal(file_scores[-1][1:], expected_scores[1:], 3)
    expected_scores = expected_scores._replace(file_id='*** OVERALL ***')
    assert global_scores.file_id == expected_scores.file_id
    assert_almost_equal(global_scores[1:], expected_scores[1:], 3)

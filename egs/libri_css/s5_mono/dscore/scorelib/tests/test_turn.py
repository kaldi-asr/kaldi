"""Tets for UEM utilities."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from intervaltree import Interval, IntervalTree

import pytest

from scorelib.turn import chop_tree, merge_turns, trim_turns, Turn
from scorelib.uem import UEM


def test_merge_turns():
    expected_turns = [
        Turn(0, 11, speaker_id='S1', file_id='FILE1'),
        Turn(0, 11, speaker_id='S2', file_id='FILE1'),
        Turn(0, 11, speaker_id='S1', file_id='FILE2'),
        ]
    turns = [
        Turn(0, 10, speaker_id='S1', file_id='FILE1'),
        Turn(9, 11, speaker_id='S1', file_id='FILE1'),
        Turn(0, 11, speaker_id='S2', file_id='FILE1'),
        Turn(0, 11, speaker_id='S1', file_id='FILE2'),
        ]
    assert set(expected_turns) == set(merge_turns(turns))


def test_trim_turns():
    turns = [
        Turn(1, 5, speaker_id='S1', file_id='FILE1'),
        Turn(6, 10, speaker_id='S1', file_id='FILE1'),
        Turn(0, 10, speaker_id='S1', file_id='FILE2'),
        ]

    # Trim with a UEM.
    uem = UEM({'FILE1' : [(2, 6), (5.8, 7)],
               'FILE2' : [(2, 3), (4, 5)]})
    expected_turns = [
        Turn(2, 5, speaker_id='S1', file_id='FILE1'),
        Turn(6, 7, speaker_id='S1', file_id='FILE1'),
        Turn(2, 3, speaker_id='S1', file_id='FILE2'),
        Turn(4, 5, speaker_id='S1', file_id='FILE2'),
        ]
    assert set(expected_turns) == set(trim_turns(turns, uem))

    # Trim without UEM.
    expected_turns = [
        Turn(2, 5, speaker_id='S1', file_id='FILE1'),
        Turn(6, 7, speaker_id='S1', file_id='FILE1'),
        Turn(2, 7, speaker_id='S1', file_id='FILE2'),
        ]
    assert set(expected_turns) == set(trim_turns(turns, None, 2, 7))


def test_chop_tree():
    def _get_tree():
        return IntervalTree.from_tuples(
            [(1, 5, 'i1'),
             (7, 10, 'i2'),
            ])

    # Interval contained within chop region.
    expected_tree = IntervalTree.from_tuples([(7, 10, 'i2')])
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 0, 6)
    assert overlap_intervals == set([Interval(1, 5, 'i1')])
    assert tree == expected_tree

    # Left overlap.
    expected_tree = IntervalTree.from_tuples([(2, 5, 'i1'), (7, 10, 'i2')])
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 0, 2)
    assert overlap_intervals == set([Interval(1, 5, 'i1')])
    assert tree == expected_tree

    # Right overlap.
    expected_tree = IntervalTree.from_tuples([(1, 5, 'i1'), (7, 9, 'i2')])
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 9, 11)
    assert overlap_intervals == set([Interval(7, 10, 'i2')])
    assert tree == expected_tree

    # Chop region contained within interval.
    expected_tree = IntervalTree.from_tuples([(1, 2, 'i1'),
                                              (3, 5, 'i1'),
                                              (7, 10, 'i2')])
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 2, 3)
    assert overlap_intervals == set([Interval(1, 5, 'i1')])
    assert tree == expected_tree

    # Overlaps two intervals.
    expected_tree = IntervalTree.from_tuples([(1, 4, 'i1'),
                                              (8, 10, 'i2')])
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 4, 8)
    assert overlap_intervals == set([Interval(1, 5, 'i1'),
                                     Interval(7, 10, 'i2')])
    assert tree == expected_tree

    # No overlap.
    expected_tree = _get_tree()
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 6, 6.5)
    assert overlap_intervals == set()
    assert tree == expected_tree

    # No trivial overlaps.
    expected_tree = _get_tree()
    tree = _get_tree()
    overlap_intervals = chop_tree(tree, 0, 1)
    assert overlap_intervals == set()
    assert tree == expected_tree
    overlap_intervals = chop_tree(tree, 10, 11)
    assert overlap_intervals == set()
    assert tree == expected_tree

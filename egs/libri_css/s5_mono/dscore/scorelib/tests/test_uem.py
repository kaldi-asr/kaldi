"""Tets for UEM utilities."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import os
import shutil
import tempfile

from numpy.testing import (assert_equal, assert_raises, assert_raises_regex)
import pytest

from scorelib.turn import Turn
from scorelib.uem import gen_uem, load_uem, write_uem, UEM


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = tempfile.mkdtemp(prefix="dscore_scorelib_test_uem__")


def test_UEM_input_validation():
    # Test type validation.
    invalid_type_msg = 'Expected sequence of pairs'
    with assert_raises_regex(TypeError, invalid_type_msg):
        UEM()['fid'] = 1
    with assert_raises_regex(TypeError, invalid_type_msg):
        UEM()['fid'] = (1, (2, 3))
    with assert_raises_regex(TypeError, invalid_type_msg):
        UEM()['fid'] = ('1', (2, 3))
    with assert_raises_regex(TypeError, invalid_type_msg):
        UEM()['fid'] = ((1,), (2, 3))
    with assert_raises_regex(TypeError, invalid_type_msg):
        UEM()['fid'] = ((3, 4, 5), (0, 1))

    # Test value validation.
    invalid_timestamp_msg = 'Could not convert interval'
    with assert_raises_regex(ValueError, invalid_timestamp_msg):
        UEM()['fid'] = (('a', 'b'), (2, 3))
    invalid_interval_msg = 'Invalid interval'
    with assert_raises_regex(ValueError, invalid_interval_msg):
        UEM()['fid'] = ((-1, 1), (2, 3))
    with assert_raises_regex(ValueError, invalid_interval_msg):
        UEM()['fid'] = ((1, 1), (2, 3))
    with assert_raises_regex(ValueError, invalid_interval_msg):
        UEM()['fid'] = ((-1, 1), (2, 3))


def test_UEM_merge_overlaps():
    overlaps_uem = UEM({'fid' : [(0, 5), (4, 10)]})
    no_overlaps_uem = UEM({'fid' : [(0, 10)]})
    assert overlaps_uem == no_overlaps_uem


def test_load_uem():
    expected_uem = UEM({
        'FILE1' : [(0, 15), (25, 30.4)],
        'FILE2' : [(4.5, 13.24)]})
    loaded_uem = load_uem(os.path.join(TEST_DIR, 'test_load.uem'))
    assert expected_uem == loaded_uem


def test_write_uem():
    uem = UEM({
        'FILE1' : [(0, 5), (4, 10.6)],
        'FILE2' : [(4.5, 13.24)]})
    tmp_uemf = os.path.join(TMP_DIR, 'test_write.uem')
    write_uem(tmp_uemf, uem)
    expected_uemf = os.path.join(TEST_DIR, 'test_write.uem')
    with open(tmp_uemf, 'rb') as f, open(expected_uemf, 'rb') as g:
        assert f.read() == g.read()


def test_gen_uem():
    ref_turns = [
        Turn(0, 14, speaker_id='SPK1', file_id='FILE1'),
        Turn(20, 25, speaker_id='SPK2', file_id='FILE1'),
        Turn(11, 45, speaker_id='SPK3', file_id='FILE2'),
    ]
    sys_turns = [
        Turn(35, 46, speaker_id='SPK1', file_id='FILE1'),
        Turn(2.5, 14, speaker_id='SPK4', file_id='FILE2'),
    ]
    expected_uem = UEM({
        'FILE1' : [(0, 46)],
        'FILE2' : [(2.5, 45)]})
    assert expected_uem == gen_uem(ref_turns, sys_turns)


def teardown_module():
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)

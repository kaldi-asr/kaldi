#! /usr/bin/env python

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
from __future__ import division
import argparse
import copy
import logging
import heapq
import sys
from collections import defaultdict

"""
This script reads 'ctm-edits' file format that is produced by align_ctm_ref.py
and modified by modify_ctm_edits.py and taint_ctm_edits.py. Its function is to
produce a segmentation and text from the ctm-edits input.

It is a milder version of the script segment_ctm_edits.py i.e. it allows
to keep more of the reference. This is useful for segmenting long-audio
based on imperfect transcripts.

The ctm-edits file format that this script expects is as follows
<file-id> <channel> <start-time> <duration> <conf> <hyp-word> <ref-word> <edit>
['tainted']
[note: file-id is really utterance-id at this point].
"""

_global_logger = logging.getLogger(__name__)
_global_logger.setLevel(logging.INFO)
_global_handler = logging.StreamHandler()
_global_handler.setLevel(logging.INFO)
_global_formatter = logging.Formatter(
    '%(asctime)s [%(pathname)s:%(lineno)s - '
    '%(funcName)s - %(levelname)s ] %(message)s')
_global_handler.setFormatter(_global_formatter)
_global_logger.addHandler(_global_handler)

_global_non_scored_words = {}


def non_scored_words():
    return _global_non_scored_words


def get_args():
    parser = argparse.ArgumentParser(
        description="""This program produces segmentation and text information
        based on reading ctm-edits input format which is produced by
        steps/cleanup/internal/get_ctm_edits.py,
        steps/cleanup/internal/modify_ctm_edits.py and
        steps/cleanup/internal/taint_ctm_edits.py.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--min-segment-length", type=float, default=0.5,
                        help="""Minimum allowed segment length (in seconds) for
                        any segment; shorter segments than this will be
                        discarded.""")
    parser.add_argument("--min-new-segment-length", type=float, default=1.0,
                        help="""Minimum allowed segment length (in seconds) for
                        newly created segments (i.e. not identical to the input
                        utterances).
                        Expected to be >= --min-segment-length.""")
    parser.add_argument("--frame-length", type=float, default=0.01,
                        help="""This only affects rounding of the output times;
                        they will be constrained to multiples of this
                        value.""")
    parser.add_argument("--max-tainted-length", type=float, default=0.05,
                        help="""Maximum allowed length of any 'tainted' line.
                        Note: 'tainted' lines may only appear at the boundary
                        of a segment""")
    parser.add_argument("--max-edge-silence-length", type=float, default=0.5,
                        help="""Maximum allowed length of silence if it appears
                        at the edge of a segment (will be truncated).  This
                        rule is relaxed if such truncation would take a segment
                        below the --min-segment-length or
                        --min-new-segment-length.""")
    parser.add_argument("--max-edge-non-scored-length", type=float,
                        default=0.5,
                        help="""Maximum allowed length of a non-scored word
                        (noise, cough, etc.) if it appears at the edge of a
                        segment (will be truncated).  This rule is relaxed if
                        such truncation would take a segment below the
                        --min-segment-length.""")
    parser.add_argument("--max-internal-silence-length", type=float,
                        default=2.0,
                        help="""Maximum allowed length of silence if it appears
                        inside a segment (will cause the segment to be
                        split).""")
    parser.add_argument("--max-internal-non-scored-length", type=float,
                        default=2.0,
                        help="""Maximum allowed length of a non-scored word
                        (noise, etc.) if it appears inside a segment (will
                        cause the segment to be split).
                        Note: reference words which are real words but OOV are
                        not included in this category.""")
    parser.add_argument("--unk-padding", type=float, default=0.05,
                        help="""Amount of padding with <unk> that we do if a
                        segment boundary is next to errors (ins, del, sub).
                        That is, we add this amount of time to the segment and
                        add the <unk> word to cover the acoustics.  If nonzero,
                        the --oov-symbol-file option must be supplied.""")
    parser.add_argument("--max-junk-proportion", type=float, default=0.1,
                        help="""Maximum proportion of the time of the segment
                        that may consist of potentially bad data, in which we
                        include 'tainted' lines of the ctm-edits input and
                        unk-padding.""")
    parser.add_argument("--min-split-point-duration", type=float, default=0.0,
                        help="""Minimum duration of silence or non-scored word
                        to be considered a viable split point when
                        truncating based on junk proportion.""")
    parser.add_argument("--max-deleted-words-kept-when-merging",
                        dest='max_deleted_words', type=int, default=1,
                        help="""When merging segments that are found to be
                        overlapping or adjacent after all other processing,
                        keep in the transcript the reference words that were
                        deleted between the segments [if any] as long as there
                        were no more than this many reference words.  Setting
                        this to zero will mean that any reference words that
                        were deleted between the segments we're about to
                        reattach will not appear in the generated transcript
                        (so we'll match the hyp).""")

    parser.add_argument("--splitting.min-silence-length",
                        dest="min_silence_length_to_split",
                        type=float, default=0.3,
                        help="""Only considers silences that are at least this
                        long as potential split points""")
    parser.add_argument("--splitting.min-non-scored-length",
                        dest="min_non_scored_length_to_split",
                        type=float, default=0.1,
                        help="""Only considers non-scored words that are at
                        least this long as potential split points""")
    parser.add_argument("--splitting.max-segment-length",
                        dest="max_segment_length_for_splitting",
                        type=float, default=10,
                        help="""Try to split long segments into segments that
                        are smaller that this size. See
                        possibly_split_long_segments() in Segment class.""")
    parser.add_argument("--splitting.hard-max-segment-length",
                        dest="hard_max_segment_length",
                        type=float, default=15,
                        help="""Split all segments that are longer than this
                        uniformly into segments of size
                        --splitting.max-segment-length""")

    parser.add_argument("--merging-score.silence-factor",
                        dest="silence_factor",
                        type=float, default=1,
                        help="""Weightage on the silence length when merging
                        segments""")
    parser.add_argument("--merging-score.incorrect-words-factor",
                        dest="incorrect_words_factor",
                        type=float, default=1,
                        help="""Weightage on the incorrect_words_length when
                        merging segments""")
    parser.add_argument("--merging-score.tainted-words-factor",
                        dest="tainted_words_factor",
                        type=float, default=1,
                        help="""Weightage on the WER including the
                        tainted words as incorrect words.""")

    parser.add_argument("--merging.max-wer",
                        dest="max_wer",
                        type=float, default=10.0,
                        help="Max WER%% of merged segments when merging")
    parser.add_argument("--merging.max-bad-proportion",
                        dest="max_bad_proportion",
                        type=float, default=0.2,
                        help="""Maximum length of silence, junk and incorrect
                        words in a merged segment allowed as a fraction of the
                        total length of merged segment.""")
    parser.add_argument("--merging.max-segment-length",
                        dest='max_segment_length_for_merging',
                        type=float, default=10,
                        help="""Maximum segment length allowed for merged
                        segment""")
    parser.add_argument("--merging.max-intersegment-incorrect-words-length",
                        dest='max_intersegment_incorrect_words_length',
                        type=float, default=0.2,
                        help="""Maximum length of intersegment region that
                        can be of incorrect word. This is to
                        allow cases where there may be a lot of silence in the
                        segment but the incorrect words are few, while
                        preventing regions that have a lot of incorrect
                        words.""")

    parser.add_argument("--oov-symbol-file", type=argparse.FileType('r'),
                        help="""Filename of file such as data/lang/oov.txt
                        which contains the text form of the OOV word, normally
                        '<unk>'.  Supplied as a file to avoid complications
                        with escaping.  Necessary if the --unk-padding option
                        has a nonzero value (which it does by default.""")
    parser.add_argument("--ctm-edits-out", type=argparse.FileType('w'),
                        help="""Filename to output an extended version of the
                        ctm-edits format with segment start and end points
                        noted.  This file is intended to be read by humans;
                        there are currently no scripts that will read it.""")
    parser.add_argument("--word-stats-out", type=argparse.FileType('w'),
                        help="""Filename for output of word-level stats, of the
                        form '<word> <bad-proportion> <total-count-in-ref>',
                        e.g. 'hello 0.12 12408', where the <bad-proportion> is
                        the proportion of the time that this reference word
                        does not make it into a segment.  It can help reveal
                        words that have problematic pronunciations or are
                        associated with transcription errors.""")

    parser.add_argument("non_scored_words_in",
                        metavar="<non-scored-words-file>",
                        type=argparse.FileType('r'),
                        help="""Filename of file containing a list of
                        non-scored words, one per line. See
                        steps/cleanup/internal/get_nonscored_words.py.""")
    parser.add_argument("ctm_edits_in", metavar="<ctm-edits-in>",
                        type=argparse.FileType('r'),
                        help="""Filename of input ctm-edits file.  Use
                        /dev/stdin for standard input.""")
    parser.add_argument("text_out", metavar="<text-out>",
                        type=argparse.FileType('w'),
                        help="""Filename of output text file (same format as
                        data/train/text, i.e.  <new-utterance-id> <word1>
                        <word2> ... <wordN>""")
    parser.add_argument("segments_out", metavar="<segments-out>",
                        type=argparse.FileType('w'),
                        help="""Filename of output segments.  This has the same
                        format as data/train/segments, but instead of
                        <recording-id>, the second field is the old
                        utterance-id, i.e <new-utterance-id> <old-utterance-id>
                        <start-time> <end-time>""")

    parser.add_argument("--verbose", type=int, default=0,
                        help="Use higher verbosity for more debugging output")

    args = parser.parse_args()

    if args.verbose > 2:
        _global_handler.setLevel(logging.DEBUG)
        _global_logger.setLevel(logging.DEBUG)

    return args


def is_tainted(split_line_of_utt):
    """Returns True if this line in ctm-edit is "tainted."""
    return len(split_line_of_utt) > 8 and split_line_of_utt[8] == 'tainted'


def compute_segment_cores(split_lines_of_utt):
    """
    This function returns a list of pairs (start-index, end-index) representing
    the cores of segments (so if a pair is (s, e), then the core of a segment
    would span (s, s+1, ... e-1).

    The argument 'split_lines_of_utt' is list of lines from a ctm-edits file
    corresponding to a single utterance.

    By the 'core of a segment', we mean a sequence of ctm-edits lines including
    at least one 'cor' line and a contiguous sequence of other lines of the
    type 'cor', 'fix' and 'sil' that must be not tainted.  The segment core
    excludes any tainted lines at the edge of a segment, which will be added
    later.

    We only initiate segments when it contains something correct and not
    realized as unk (i.e. ref==hyp); and we extend it with anything that is
    'sil' or 'fix' or 'cor' that is not tainted.  Contiguous regions of 'true'
    in the resulting boolean array will then become the cores of prototype
    segments, and we'll add any adjacent tainted words (or parts of them).
    """
    num_lines = len(split_lines_of_utt)
    line_is_in_segment_core = [False] * num_lines
    # include only the correct lines
    for i in range(num_lines):
        if (split_lines_of_utt[i][7] == 'cor'
                and split_lines_of_utt[i][4] == split_lines_of_utt[i][6]):
            line_is_in_segment_core[i] = True

    # extend each proto-segment forwards as far as we can:
    for i in range(1, num_lines):
        if line_is_in_segment_core[i - 1] and not line_is_in_segment_core[i]:
            edit_type = split_lines_of_utt[i][7]
            if (not is_tainted(split_lines_of_utt[i])
                    and (edit_type == 'cor' or edit_type == 'sil'
                         or edit_type == 'fix')):
                line_is_in_segment_core[i] = True

    # extend each proto-segment backwards as far as we can:
    for i in reversed(range(0, num_lines - 1)):
        if line_is_in_segment_core[i + 1] and not line_is_in_segment_core[i]:
            edit_type = split_lines_of_utt[i][7]
            if (not is_tainted(split_lines_of_utt[i])
                    and (edit_type == 'cor' or edit_type == 'sil'
                         or edit_type == 'fix')):
                line_is_in_segment_core[i] = True

    # Get contiguous regions of line in the form of a list
    # of (start_index, end_index)
    segment_ranges = []
    cur_segment_start = None
    for i in range(0, num_lines):
        if line_is_in_segment_core[i]:
            if cur_segment_start is None:
                cur_segment_start = i
        else:
            if cur_segment_start is not None:
                segment_ranges.append((cur_segment_start, i))
                cur_segment_start = None
    if cur_segment_start is not None:
        segment_ranges.append((cur_segment_start, num_lines))

    return segment_ranges


class SegmentStats(object):
    """Class to store various statistics of segments."""

    def __init__(self):
        self.num_incorrect_words = 0
        self.num_tainted_words = 0
        self.incorrect_words_length = 0
        self.tainted_nonsilence_length = 0
        self.silence_length = 0
        self.num_words = 0
        self.total_length = 0

    def wer(self):
        """Returns WER%"""
        try:
            return float(self.num_incorrect_words) * 100.0 / self.num_words
        except ZeroDivisionError:
            return float("inf")

    def bad_proportion(self):
        assert self.total_length > 0
        proportion = float(self.silence_length + self.tainted_nonsilence_length
                           + self.incorrect_words_length) / self.total_length
        if proportion > 1.00005:
            raise RuntimeError("Error in segment stats {0}".format(self))
        return proportion

    def incorrect_proportion(self):
        assert self.total_length > 0
        proportion = float(self.incorrect_words_length) / self.total_length
        if proportion > 1.00005:
            raise RuntimeError("Error in segment stats {0}".format(self))
        return proportion

    def combine(self, other, scale=1):
        """Merges this stats with another stats object."""
        self.num_incorrect_words += scale * other.num_incorrect_words
        self.num_tainted_words += scale * other.num_tainted_words
        self.num_words += scale * other.num_words
        self.incorrect_words_length += scale * other.incorrect_words_length
        self.tainted_nonsilence_length += (scale
                                           * other.tainted_nonsilence_length)
        self.silence_length += scale * other.silence_length
        self.total_length += scale * other.total_length

    def assert_equal(self, other):
        try:
            assert self.num_incorrect_words == other.num_incorrect_words
            assert self.num_tainted_words == other.num_tainted_words
            assert (abs(self.incorrect_words_length
                        - other.incorrect_words_length) < 0.01)
            assert (abs(self.tainted_nonsilence_length
                        - other.tainted_nonsilence_length) < 0.01)
            assert abs(self.silence_length - other.silence_length) < 0.01
            assert self.num_words == other.num_words
            assert abs(self.total_length - other.total_length) < 0.01
        except AssertionError:
            _global_logger.error("self %s != other %s", self, other)
            raise

    def compare(self, other):
        """Returns true if this stats is same as another stats object."""
        if self.num_incorrect_words != other.num_incorrect_words:
            return False
        if self.num_tainted_words != other.num_tainted_words:
            return False
        if self.incorrect_words_length != other.incorrect_words_length:
            return False
        if self.tainted_nonsilence_length != other.tainted_nonsilence_length:
            return False
        if self.silence_length != other.silence_length:
            return False
        if self.num_words != other.num_words:
            return False
        if self.total_length != other.total_length:
            return False
        return True

    def __str__(self):
        return ("num-incorrect-words={num_incorrect:d},"
                "num-tainted-words={num_tainted:d},"
                "num-words={num_words:d},"
                "incorrect-length={incorrect_length:.2f},"
                "silence-length={sil_length:.2f},"
                "tainted-nonsilence-length={tainted_nonsilence_length:.2f},"
                "total-length={total_length:.2f}".format(
                    num_incorrect=self.num_incorrect_words,
                    num_tainted=self.num_tainted_words,
                    num_words=self.num_words,
                    incorrect_length=self.incorrect_words_length,
                    sil_length=self.silence_length,
                    tainted_nonsilence_length=self.tainted_nonsilence_length,
                    total_length=self.total_length))


class Segment(object):
    """Class to store segments."""

    def __init__(self, split_lines_of_utt, start_index, end_index,
                 debug_str=None, compute_segment_stats=False,
                 segment_stats=None):
        self.split_lines_of_utt = split_lines_of_utt

        # start_index is the index of the first line that appears in this
        # segment, and end_index is one past the last line.  This does not
        # include unk-padding.
        self.start_index = start_index
        self.end_index = end_index
        assert end_index > start_index

        # If the following values are nonzero, then when we create the segment
        # we will add <unk> at the start and end of the segment [representing
        # partial words], with this amount of additional audio.
        self.start_unk_padding = 0.0
        self.end_unk_padding = 0.0

        # debug_str keeps track of the 'core' of the segment.
        if debug_str is None:
            debug_str = 'core-start={0},core-end={1}'.format(start_index,
                                                             end_index)
        else:
            assert type(debug_str) == str
        self.debug_str = debug_str

        # This gives the proportion of the time of the first line in the
        # segment that we keep.  Usually 1.0 but may be less if we've trimmed
        # away some proportion of the time.
        self.start_keep_proportion = 1.0
        # This gives the proportion of the time of the last line in the segment
        # that we keep.  Usually 1.0 but may be less if we've trimmed away some
        # proportion of the time.
        self.end_keep_proportion = 1.0

        self.stats = None

        if compute_segment_stats:
            self.compute_stats()

        if segment_stats is not None:
            self.compute_stats()
            self.stats.assert_equal(segment_stats)
            self.stats = segment_stats

    def copy(self, copy_stats=True):
        segment = Segment(self.split_lines_of_utt, self.start_index,
                          self.end_index, debug_str=self.debug_str,
                          segment_stats=(None if not copy_stats
                                         else copy.deepcopy(self.stats)))
        segment.start_keep_proportion = self.start_keep_proportion
        segment.end_keep_proportion = self.end_keep_proportion
        segment.start_unk_padding = self.start_unk_padding
        segment.end_unk_padding = self.end_unk_padding
        return segment

    def __str__(self):
        return self.debug_info()

    def compute_stats(self):
        """Compute stats for this segment and store them in SegmentStats
        structure.
        This is typically called just before merging segments.
        """
        self.stats = SegmentStats()
        for i in range(self.start_index, self.end_index):
            this_duration = float(self.split_lines_of_utt[i][3])
            assert self.start_keep_proportion == 1.0
            assert self.end_keep_proportion == 1.0
            # TODO(vimal): Decide if keep proportion must be applied
            # if i == self.start_index:
            #     this_duration *= self.start_keep_proportion
            # if i == self.end_index - 1:
            #     this_duration *= self.end_keep_proportion
            if self.end_index - 1 == self.start_index:
                # TODO(vimal): Is this true?
                assert self.start_keep_proportion == self.end_keep_proportion

            try:
                if self.split_lines_of_utt[i][7] not in ['cor', 'fix', 'sil']:
                    # TODO(vimal): The commented part below is is apparently
                    # not true in modify_ctm_edits.py.
                    # Need to check this or change comments there.
                    # assert (self.split_lines_of_utt[i][6]
                    #         not in non_scored_words)
                    assert not is_tainted(self.split_lines_of_utt[i])
                    self.stats.num_incorrect_words += 1
                    self.stats.incorrect_words_length += this_duration
                if self.split_lines_of_utt[i][7] == 'sil':
                    self.stats.silence_length += this_duration
                else:
                    if (self.split_lines_of_utt[i][6]
                            not in non_scored_words()):
                        self.stats.num_words += 1
                if (is_tainted(self.split_lines_of_utt[i])
                        and self.split_lines_of_utt[i][7] not in 'sil'
                        and (self.split_lines_of_utt[i][6]
                             not in non_scored_words())):
                    # If ref_word is not a non-scored word, this would be
                    # counted as an incorrect word.
                    self.stats.num_tainted_words += 1
                    self.stats.tainted_nonsilence_length += this_duration
            except Exception:
                _global_logger.error(
                    "Something went wrong when computing stats at "
                    "ctm line %s", self.split_lines_of_utt[i])
                raise
        self.stats.total_length = self.length()

        try:
            assert (self.stats.tainted_nonsilence_length
                    + self.stats.silence_length
                    + self.stats.incorrect_words_length - 0.001
                    <= self.stats.total_length)
        except AssertionError:
            _global_logger.error(
                "Something wrong with the stats for segment %s", self)
            raise

    def possibly_add_tainted_lines(self):
        """
        This is stage 1 of segment processing (after creating the boundaries of
        the core of the segment, which is done outside of this class).

        This function may reduce start_index and/or increase end_index by
        including a single adjacent 'tainted' line from the ctm-edits file.
        This is only done if the lines at the boundaries of the segment are
        currently real non-silence words and not non-scored words.  The idea is
        that we probably don't want to start or end the segment right at the
        boundary of a real word, we want to add some kind of padding.
        """
        split_lines_of_utt = self.split_lines_of_utt
        # we're iterating over the segment (start, end)
        for b in [False, True]:
            if b:
                boundary_index = self.end_index - 1
                adjacent_index = self.end_index
            else:
                boundary_index = self.start_index
                adjacent_index = self.start_index - 1
            if (adjacent_index >= 0
                    and adjacent_index < len(split_lines_of_utt)):
                # only consider merging the adjacent word into the segment if
                # we're not at the boundary of the utterance.
                adjacent_line_is_tainted = is_tainted(
                    split_lines_of_utt[adjacent_index])
                # if the adjacent line wasn't tainted, then there must have
                # been another stronger reason why we didn't include it in the
                # core of the segment (probably that it was an ins, del or
                # sub), so there is no point considering it.
                if adjacent_line_is_tainted:
                    boundary_edit_type = split_lines_of_utt[boundary_index][7]
                    boundary_ref_word = split_lines_of_utt[boundary_index][6]
                    # Even if the edit_type is 'cor', it is possible that
                    # column 4 (hyp_word) is not the same as column 6
                    # (ref_word) because the ref_word is an OOV and the
                    # hyp_word is OOV symbol.

                    # we only add the tainted line to the segment if the word
                    # at the boundary was a non-silence word that was correctly
                    # decoded and not fixed [see modify_ctm_edits.py.]
                    if (boundary_edit_type == 'cor'
                            and (boundary_ref_word
                                 not in non_scored_words())):
                        # Add the adjacent tainted line to the segment.
                        if b:
                            self.end_index += 1
                        else:
                            self.start_index -= 1

    def possibly_split_segment(self, max_internal_silence_length,
                               max_internal_non_scored_length):
        """
        This is stage 3 of segment processing.
        This function will split a segment into multiple pieces if any of the
        internal [non-boundary] silences or non-scored words are longer
        than the allowed values --max-internal-silence-length and
        --max-internal-non-scored-length.
        This function returns a list of segments.
        In the normal case (where there is no splitting) it just returns an
        array with a single element 'self'.

        Note: --max-internal-silence-length and
        --max-internal-non-scored-length can be set to very large values
        to avoid any splitting.
        """
        # make sure the segment hasn't been processed more than we expect.
        assert (self.start_unk_padding == 0.0 and self.end_unk_padding == 0.0
                and self.start_keep_proportion == 1.0
                and self.end_keep_proportion == 1.0)
        segments = []  # the answer
        cur_start_index = self.start_index
        cur_start_is_split = False
        # only consider splitting at non-boundary lines.  [we'd just truncate
        # the boundary lines.]
        for index_to_split_at in range(cur_start_index + 1,
                                       self.end_index - 1):
            this_split_line = self.split_lines_of_utt[index_to_split_at]
            this_duration = float(this_split_line[3])
            this_edit_type = this_split_line[7]
            this_ref_word = this_split_line[6]
            if ((this_edit_type == 'sil' and
                 this_duration > max_internal_silence_length)
                    or (this_ref_word in non_scored_words()
                        and (this_duration
                             > max_internal_non_scored_length))):
                # We split this segment at this index, dividing the word in two
                # [later on, in possibly_truncate_boundaries, it may be further
                # truncated.]
                # Note: we use 'index_to_split_at + 1' because the Segment
                # constructor takes an 'end-index' which is interpreted as one
                # past the end.
                new_segment = Segment(self.split_lines_of_utt, cur_start_index,
                                      index_to_split_at + 1,
                                      debug_str=self.debug_str)
                if cur_start_is_split:
                    new_segment.start_keep_proportion = 0.5
                new_segment.end_keep_proportion = 0.5
                cur_start_is_split = True
                cur_start_index = index_to_split_at
                segments.append(new_segment)
        if len(segments) == 0:  # We did not split.
            segments.append(self)
        else:
            # We did split.  Add the very last segment.
            new_segment = Segment(self.split_lines_of_utt, cur_start_index,
                                  self.end_index,
                                  debug_str=self.debug_str)
            assert cur_start_is_split
            new_segment.start_keep_proportion = 0.5
            segments.append(new_segment)
        return segments

    def possibly_split_long_segment(self, max_segment_length,
                                    hard_max_segment_length,
                                    min_silence_length_to_split,
                                    min_non_scored_length_to_split):
        """
        This is stage 4 of segment processing.
        This function will split a segment into multiple pieces if it is
        longer than the value --max-segment-length.
        It tries to split at silences and non-scored words that are
        at least --min-silence-length-to-split or
        --min-non-scored-length-to-split long.
        If this is not possible and the segments are still longer than
        --hard-max-segment-length, then this is split into equal length
        pieces of approximately --max-segment-length long.
        This function returns a list of segments.
        In the normal case (where there is no splitting) it just returns an
        array with a single element 'self'.
        """
        # make sure the segment hasn't been processed more than we expect.
        assert self.start_unk_padding == 0.0 and self.end_unk_padding == 0.0
        if self.length() < max_segment_length:
            return [self]

        segments = [self]  # the answer
        cur_start_index = self.start_index

        split_indexes = []
        # only consider splitting at non-boundary lines.  [we'd just truncate
        # the boundary lines.]
        for index_to_split_at in range(cur_start_index + 1,
                                       self.end_index - 1):
            this_split_line = self.split_lines_of_utt[index_to_split_at]
            this_duration = float(this_split_line[3])
            this_edit_type = this_split_line[7]
            this_ref_word = this_split_line[6]
            this_is_tainted = is_tainted(this_split_line)
            if (this_edit_type == 'sil'
                    and this_duration > min_silence_length_to_split):
                split_indexes.append((index_to_split_at, this_duration,
                                      this_is_tainted))

            if (this_ref_word in non_scored_words()
                    and (this_duration > min_non_scored_length_to_split)):
                split_indexes.append((index_to_split_at, this_duration,
                                      this_is_tainted))
        split_indexes.sort(key=lambda x: x[1], reverse=True)
        split_indexes.sort(key=lambda x: x[2])

        while True:
            if len(split_indexes) == 0:
                break

            new_segments = []

            for segment in segments:
                if segment.length() < max_segment_length:
                    new_segments.append(segment)
                    continue

                try:
                    index_to_split_at = next(
                        (x[0] for x in split_indexes
                         if (x[0] > segment.start_index
                             and x[0] < segment.end_index - 1)))
                except StopIteration:
                    _global_logger.debug(
                        "Could not find an index in the range (%d, %d) in "
                        "split-indexes %s", segment.start_index,
                        segment.end_index - 1, split_indexes)
                    new_segments.append(segment)
                    continue

                # We split this segment at this index, dividing the word in two
                # [later on, in possibly_truncate_boundaries, it may be further
                # truncated.]
                # Note: we use 'index_to_split_at + 1' because the Segment
                # constructor takes an 'end-index' which is interpreted as one
                # past the end.
                new_segment = Segment(
                    self.split_lines_of_utt, segment.start_index,
                    index_to_split_at + 1, debug_str=self.debug_str)
                new_segment.end_keep_proportion = 0.5
                new_segments.append(new_segment)

                new_segment = Segment(
                    self.split_lines_of_utt, index_to_split_at,
                    segment.end_index, debug_str=self.debug_str)
                new_segment.start_keep_proportion = 0.5
                new_segments.append(new_segment)

            if len(segments) == len(new_segments):
                # No splitting done
                break
            segments = new_segments

            for i, x in enumerate(segments):
                _global_logger.debug("Segment %d = %s", i, x)

        new_segments = []
        # Split segments that are still very long
        for segment in segments:
            if segment.length() < hard_max_segment_length:
                new_segments.append(segment)
                continue

            cur_start_index = segment.start_index
            cur_start = segment.start_time()

            index_to_split_at = None
            try:
                while True:
                    index_to_split_at = next(
                        (i for i in range(cur_start_index, segment.end_index)
                         if (float(self.split_lines_of_utt[i][2])
                             >= cur_start + max_segment_length)))

                    new_segment = Segment(
                        self.split_lines_of_utt, cur_start_index,
                        index_to_split_at)
                    new_segments.append(new_segment)

                    cur_start_index = index_to_split_at
                    cur_start = float(
                        self.split_lines_of_utt[cur_start_index][2])
                    index_to_split_at = None

                    if (segment.end_time() - cur_start
                            < hard_max_segment_length):
                        raise StopIteration
            except StopIteration:
                if index_to_split_at is None:
                    _global_logger.debug(
                        "Could not find an index in the range (%d, %d) with "
                        "start time > %.2f", cur_start_index,
                        segment.end_index, cur_start + max_segment_length)
                new_segment = Segment(
                    self.split_lines_of_utt, cur_start_index,
                    segment.end_index)
                new_segments.append(new_segment)
                break
        segments = new_segments
        return segments

    def possibly_truncate_boundaries(self, max_edge_silence_length,
                                     max_edge_non_scored_length):
        """
        This is stage 5 of segment processing.
        It will truncate the silences and non-scored words at the segment
        boundaries if they are longer than the --max-edge-silence-length and
        --max-edge-non-scored-length respectively
        (and to the extent that this wouldn't take us below the
        --min-segment-length or --min-new-segment-length. See
        relax_boundary_truncation()).

        Note: --max-edge-silence-length and --max-edge-non-scored-length
        can be set to very large values to avoid any truncation.
        """
        for b in [True, False]:
            if b:
                this_index = self.start_index
            else:
                this_index = self.end_index - 1
            this_split_line = self.split_lines_of_utt[this_index]
            truncated_duration = None
            this_duration = float(this_split_line[3])
            this_edit = this_split_line[7]
            this_ref_word = this_split_line[6]
            if (this_edit == 'sil'
                    and this_duration > max_edge_silence_length):
                truncated_duration = max_edge_silence_length
            elif (this_ref_word in non_scored_words()
                  and this_duration > max_edge_non_scored_length):
                truncated_duration = max_edge_non_scored_length
            if truncated_duration is not None:
                keep_proportion = truncated_duration / this_duration
                if b:
                    self.start_keep_proportion = keep_proportion
                else:
                    self.end_keep_proportion = keep_proportion

    def relax_boundary_truncation(self, min_segment_length,
                                  min_new_segment_length):
        """
        This relaxes the segment-boundary truncation of
        possibly_truncate_boundaries(), if it would take us below
        min-new-segment-length or min-segment-length.

        Note: this does not relax the boundary truncation for a particular
        boundary (start or end) if that boundary corresponds to a 'tainted'
        line of the ctm (because it's dangerous to include too much 'tainted'
        audio).
        """
        # this should be called before adding unk padding.
        assert self.start_unk_padding == self.end_unk_padding == 0.0
        if self.start_keep_proportion == self.end_keep_proportion == 1.0:
            return  # nothing to do there was no truncation.
        length_cutoff = max(min_new_segment_length, min_segment_length)
        length_with_truncation = self.length()
        if length_with_truncation >= length_cutoff:
            return  # Nothing to do.
        orig_start_keep_proportion = self.start_keep_proportion
        orig_end_keep_proportion = self.end_keep_proportion
        if not is_tainted(self.split_lines_of_utt[self.start_index]):
            self.start_keep_proportion = 1.0
        if not is_tainted(self.split_lines_of_utt[self.end_index - 1]):
            self.end_keep_proportion = 1.0
        length_with_relaxed_boundaries = self.length()
        if length_with_relaxed_boundaries <= length_cutoff:
            # Completely undo the truncation [to the extent allowed by the
            # presence of tainted lines at the start/end] if, even without
            # truncation, we'd be below the length cutoff.  This segment may be
            # removed later on (but it may not, if removing truncation makes us
            # identical to the input utterance, and the length is between
            # min_segment_length min_new_segment_length).
            return
        # Next, compute an interpolation constant a such that the
        # {start,end}_keep_proportion values will equal
        # a
        # * [values-computed-by-possibly_truncate_boundaries()]
        # + (1-a) * [completely-relaxed-values].
        # we're solving the equation:
        # length_cutoff = a * length_with_truncation
        #                 + (1-a) * length_with_relaxed_boundaries
        # -> length_cutoff - length_with_relaxed_boundaries =
        #        a * (length_with_truncation - length_with_relaxed_boundaries)
        # -> a = (length_cutoff - length_with_relaxed_boundaries)
        #        / (length_with_truncation - length_with_relaxed_boundaries)
        a = (length_cutoff - length_with_relaxed_boundaries) / (length_with_truncation - length_with_relaxed_boundaries)
        if a < 0.0 or a > 1.0:
            # TODO(vimal): Should this be an error?
            _global_logger.warn("bad 'a' value = %.4f", a)
            return
        self.start_keep_proportion = (
            a * orig_start_keep_proportion
            + (1 - a) * self.start_keep_proportion)
        self.end_keep_proportion = (
            a * orig_end_keep_proportion + (1 - a) * self.end_keep_proportion)
        if abs(self.length() - length_cutoff) >= 0.01:
            # TODO(vimal): Should this be an error?
            _global_logger.warn(
                "possible problem relaxing boundary "
                "truncation, length is %.2f vs %.2f", self.length(),
                length_cutoff)

    def possibly_add_unk_padding(self, max_unk_padding):
        """
        This is stage 7 of segment processing.
        This function may set start_unk_padding and end_unk_padding to nonzero
        values.  This is done if the current boundary words are real, scored
        words and we're not next to the beginning or end of the utterance.
        """
        for b in [True, False]:
            if b:
                this_index = self.start_index
            else:
                this_index = self.end_index - 1
            this_split_line = self.split_lines_of_utt[this_index]
            this_start_time = float(this_split_line[2])
            this_ref_word = this_split_line[6]
            this_edit = this_split_line[7]
            if this_edit == 'cor' and this_ref_word not in non_scored_words():
                # we can consider adding unk-padding.
                if b:   # start of utterance.
                    unk_padding = max_unk_padding
                    # close to beginning of file
                    if unk_padding > this_start_time:
                        unk_padding = this_start_time
                    # If we could add less than half of the specified
                    # unk-padding, don't add any (because when we add
                    # unk-padding we add the unknown-word symbol '<unk>', and
                    # if there isn't enough space to traverse the HMM we don't
                    # want to do it at all.
                    if unk_padding < 0.5 * max_unk_padding:
                        unk_padding = 0.0
                    self.start_unk_padding = unk_padding
                else:   # end of utterance.
                    this_end_time = this_start_time + float(this_split_line[3])
                    last_line = self.split_lines_of_utt[-1]
                    utterance_end_time = (float(last_line[2])
                                          + float(last_line[3]))
                    max_allowable_padding = utterance_end_time - this_end_time
                    assert max_allowable_padding > -0.01
                    unk_padding = max_unk_padding
                    if unk_padding > max_allowable_padding:
                        unk_padding = max_allowable_padding
                    # If we could add less than half of the specified
                    # unk-padding, don't add any (because when we add
                    # unk-padding we add the unknown-word symbol '<unk>',
                    # and if there isn't enough space to traverse the HMM we
                    # don't want to do it at all.
                    if unk_padding < 0.5 * max_unk_padding:
                        unk_padding = 0.0
                    self.end_unk_padding = unk_padding

    def start_time(self):
        """Returns the start time of the utterance (within the enclosing
        utterance).
        This is before any rounding.
        """
        if self.start_index == len(self.split_lines_of_utt):
            assert self.end_index == len(self.split_lines_of_utt)
            return self.end_time()
        first_line = self.split_lines_of_utt[self.start_index]
        first_line_start = float(first_line[2])
        first_line_duration = float(first_line[3])
        first_line_end = first_line_start + first_line_duration
        return (first_line_end - self.start_unk_padding
                - (first_line_duration * self.start_keep_proportion))

    def debug_info(self, include_stats=True):
        """Returns some string-valued information about 'this' that is useful
        for debugging."""
        if include_stats and self.stats is not None:
            stats = 'wer={wer:.2f},{stats},'.format(
                wer=self.stats.wer(), stats=self.stats)
        else:
            stats = ''

        return ('start={start:d},end={end:d},'
                'unk-padding={start_unk_padding:.2f},{end_unk_padding:.2f},'
                'keep-proportion={start_prop:.2f},{end_prop:.2f},'
                'start-time={start_time:.2f},end-time={end_time:.2f},'
                '{stats}'
                'debug-str={debug_str}'.format(
                    start=self.start_index, end=self.end_index,
                    start_unk_padding=self.start_unk_padding,
                    end_unk_padding=self.end_unk_padding,
                    start_prop=self.start_keep_proportion,
                    end_prop=self.end_keep_proportion,
                    start_time=self.start_time(), end_time=self.end_time(),
                    stats=stats, debug_str=self.debug_str))

    def end_time(self):
        """Returns the start time of the utterance (within the enclosing
        utterance)."""
        if self.end_index == 0:
            assert self.start_index == 0
            return self.start_time()
        last_line = self.split_lines_of_utt[self.end_index - 1]
        last_line_start = float(last_line[2])
        last_line_duration = float(last_line[3])
        return (last_line_start
                + (last_line_duration * self.end_keep_proportion)
                + self.end_unk_padding)

    def length(self):
        """Returns the segment length in seconds."""
        return self.end_time() - self.start_time()

    def is_whole_utterance(self):
        """returns true if this segment corresponds to the whole utterance that
        it's a part of (i.e. its start/end time are zero and the end-time of
        the last segment."""
        last_line_of_utt = self.split_lines_of_utt[-1]
        last_line_end_time = (float(last_line_of_utt[2])
                              + float(last_line_of_utt[3]))
        return (abs(self.start_time() - 0.0) < 0.001
                and abs(self.end_time() - last_line_end_time) < 0.001)

    def get_junk_proportion(self):
        """Returns the proportion of the duration of this segment that consists
        of unk-padding and tainted lines of input (will be between 0.0 and
        1.0)."""
        # Note: only the first and last lines could possibly be tainted as
        # that's how we create the segments; and if either or both are tainted
        # the utterance must contain other lines, so double-counting is not a
        # problem.
        junk_duration = self.start_unk_padding + self.end_unk_padding
        first_split_line = self.split_lines_of_utt[self.start_index]
        if is_tainted(first_split_line):
            first_duration = float(first_split_line[3])
            junk_duration += first_duration * self.start_keep_proportion
        last_split_line = self.split_lines_of_utt[self.end_index - 1]
        if is_tainted(last_split_line):
            last_duration = float(last_split_line[3])
            junk_duration += last_duration * self.end_keep_proportion
        return junk_duration / self.length()

    def get_junk_duration(self):
        """Returns duration of junk"""
        return self.get_junk_proportion() * self.length()

    def merge_adjacent_segment(self, other):
        """
        This function will merge the segment in 'other' with the segment
        in 'self'.  It is only to be called when 'self' and 'other' are from
        the same utterance, 'other' is after 'self' in time order (based on
        the original segment cores), and self.end_index <= self.start_index
        i.e. the two segments might have at most one index in common,
        which is usually a tainted word or silence.
        """
        try:
            assert self.end_index <= other.start_index + 1
            assert self.start_time() < other.end_time()
            assert self.split_lines_of_utt is other.split_lines_of_utt
        except AssertionError:
            _global_logger.error("self: %s", self)
            _global_logger.error("other: %s", other)
            raise

        assert self.start_index == 0 or self.start_index != other.start_index

        _global_logger.debug("Before merging: %s", self)

        assert not self.stats.compare(other.stats), "%s %s" % (self, other)
        self.stats.combine(other.stats)

        if self.end_index == other.start_index + 1:
            overlapping_segment = Segment(
                self.split_lines_of_utt, other.start_index,
                self.end_index, compute_segment_stats=True)
            self.stats.combine(overlapping_segment.stats, scale=-1)

        _global_logger.debug("Other segment: %s", other)

        self.debug_str = "({0}/merged-with-adjacent/{1})".format(
            self.debug_str, other.debug_str)

        # everything that relates to the end of this segment gets copied
        # from 'other'.
        self.end_index = other.end_index
        self.end_unk_padding = other.end_unk_padding
        self.end_keep_proportion = other.end_keep_proportion

        _global_logger.debug("After merging %s", self)
        return

    def merge_with_segment(self, other, max_deleted_words):
        """
        This function will merge the segment in 'other' with the segment
        in 'self'.  It is only to be called when 'self' and 'other' are from
        the same utterance, 'other' is after 'self' in time order (based on
        the original segment cores), and self.end_time() >= other.start_time().
        Note: in this situation there will normally be deleted words
        between the two segments.  What this program does with the deleted
        words depends on '--max-deleted-words-kept-when-merging'.  If there
        were any inserted words in the transcript (less likely), this
        program will keep the reference.

        Note: --max-deleted-words-kept-when-merging can be set to a very
        large value to keep all the words.
        """
        try:
            assert self.end_time() >= other.start_time()
            assert self.start_time() < other.end_time()
            assert self.split_lines_of_utt is other.split_lines_of_utt
        except AssertionError:
            _global_logger.error("self: %s", self)
            _global_logger.error("other: %s", other)
            raise

        assert self.start_index == 0 or self.start_index != other.start_index

        _global_logger.debug("Before merging: %s", self)

        assert (not self.stats.compare(other.stats)
                or self.start_time() != other.start_time()
                or self.end_time() != other.end_time()
                ), "%s %s" % (self, other)
        self.stats.combine(other.stats)

        _global_logger.debug("Other segment: %s", other)

        orig_self_end_index = self.end_index
        self.debug_str = "({0}/merged-with/{1})".format(
            self.debug_str, other.debug_str)

        # everything that relates to the end of this segment gets copied
        # from 'other'.
        self.end_index = other.end_index
        self.end_unk_padding = other.end_unk_padding
        self.end_keep_proportion = other.end_keep_proportion

        _global_logger.debug("After merging %s", self)

        # The next thing we have to do is to go over any lines of the ctm that
        # appear between 'self' and 'other', or are shared between both (this
        # would only happen for tainted silence or non-scored-word segments),
        # and decide what to do with them.  We'll keep the reference for any
        # substitutions or insertions (which anyway are unlikely to appear
        # in these merged segments).  Note: most of this happens in
        # self.Text(), but at this point we need to decide whether to mark any
        # deletions as 'discard-this-word'.
        try:
            if orig_self_end_index <= other.start_index:
                # No overlap in indexes
                first_index_of_overlap = orig_self_end_index
                last_index_of_overlap = other.start_index - 1
                segment = Segment(
                    self.split_lines_of_utt, orig_self_end_index,
                    other.start_index, compute_segment_stats=True)
                self.stats.combine(segment.stats)
            else:
                first_index_of_overlap = other.start_index
                last_index_of_overlap = orig_self_end_index - 1

            num_deleted_words = 0
            for i in range(first_index_of_overlap, last_index_of_overlap + 1):
                edit_type = self.split_lines_of_utt[i][7]
                if edit_type == 'del':
                    num_deleted_words += 1
            if num_deleted_words > max_deleted_words:
                for i in range(first_index_of_overlap,
                               last_index_of_overlap + 1):
                    if self.split_lines_of_utt[i][7] == 'del':
                        self.split_lines_of_utt[i].append(
                            'do-not-include-in-text')
        except:
            _global_logger.error(
                "first-index-of-overlap = %d", first_index_of_overlap)
            _global_logger.error(
                "last-index-of-overlap = %d", last_index_of_overlap)
            _global_logger.error("line = %d = %s", i,
                                 self.split_lines_of_utt[i])
            raise
        _global_logger.debug("After merging %s", self)

    def contains_atleast_one_scored_non_oov_word(self):
        """
        this will return true if there is at least one word in the utterance
        that's a scored word (not a non-scored word) and not an OOV word that's
        realized as unk.  This becomes a filter on keeping segments.
        """
        for i in range(self.start_index, self.end_index):
            this_split_line = self.split_lines_of_utt[i]
            this_hyp_word = this_split_line[4]
            this_ref_word = this_split_line[6]
            this_edit = this_split_line[7]
            if (this_edit == 'cor' and this_ref_word not in non_scored_words()
                    and this_ref_word == this_hyp_word):
                return True
        return False

    def text(self, oov_symbol, eps_symbol="<eps_symbol>"):
        """Returns the text corresponding to this utterance, as a string."""
        text_array = []
        if self.start_unk_padding != 0.0:
            text_array.append(oov_symbol)
        for i in range(self.start_index, self.end_index):
            this_split_line = self.split_lines_of_utt[i]
            this_ref_word = this_split_line[6]
            if (this_ref_word != eps_symbol
                    and this_split_line[-1] != 'do-not-include-in-text'):
                text_array.append(this_ref_word)
        if self.end_unk_padding != 0.0:
            text_array.append(oov_symbol)
        return ' '.join(text_array)


class SegmentsMerger(object):
    """This class contains methods for merging segments. It stores the
    appropriate statistics required for this process in objects of
    SegmentStats class.

    Paramters:
        segments - a reference to the list of inital segments
        merged_segments - stores all the initial segments as well
                          as the newly created segments
        between_segments - stores the inter-segment "segments"
                           for the initial segments
        split_lines_of_utt - a reference to the CTM lines
    """

    def __init__(self, segments):
        self.segments = segments

        try:
            self.split_lines_of_utt = segments[0].split_lines_of_utt
        except IndexError as e:
            _global_logger.error("No input segments found!")
            raise e

        self.merged_segments = {}
        self.between_segments = [None for i in range(len(segments) + 1)]

        if segments[0].start_index > 0:
            self.between_segments[0] = Segment(
                self.split_lines_of_utt, 0, segments[0].start_index,
                compute_segment_stats=True)

        for i, x in enumerate(segments):
            x.compute_stats()
            self.merged_segments[(i, )] = x

            if i > 0 and segments[i].start_index > segments[i - 1].end_index:
                self.between_segments[i] = Segment(
                    self.split_lines_of_utt, segments[i - 1].end_index,
                    segments[i].start_index, compute_segment_stats=True)

        if segments[-1].end_index < len(self.split_lines_of_utt):
            self.between_segments[-1] = Segment(
                self.split_lines_of_utt, segments[-1].end_index,
                len(self.split_lines_of_utt), compute_segment_stats=True)

    def _get_merged_cluster(self, cluster1, cluster2, rejected_clusters=None,
                            max_intersegment_incorrect_words_length=1):
        try:
            assert cluster2[0] > cluster1[-1]
            new_cluster = cluster1 + cluster2
            new_cluster_tup = tuple(new_cluster)

            if (rejected_clusters is not None
                    and new_cluster_tup in rejected_clusters):
                return (None, new_cluster, True)

            if new_cluster_tup in self.merged_segments:
                return (self.merged_segments[new_cluster_tup],
                        new_cluster, False)

            if cluster1[-1] == -1:
                assert len(cluster1) == 1
                # Consider merging cluster2 with the region before the 0^th
                # segment
                if (self.between_segments[0] is None
                        or self.between_segments[0].stats.total_length == 0
                        or (self.between_segments[0]
                            .stats.incorrect_words_length
                            > max_intersegment_incorrect_words_length)):
                    # Reject zero length or bad start region
                    return (None, new_cluster, True)
                merged_segment = self.between_segments[0].copy()
            else:
                merged_segment = self.merged_segments[tuple(cluster1)].copy()

                if cluster2[0] == len(self.segments):
                    assert len(cluster2) == 1
                    if (self.between_segments[-1] is None
                            or (self.between_segments[-1]
                                .stats.total_length == 0)
                            or (self.between_segments[-1]
                                .stats.incorrect_words_length
                                > max_intersegment_incorrect_words_length)):
                        # Reject zero length or bad end region
                        return (None, new_cluster, True)
                if self.between_segments[cluster2[0]] is not None:
                    if (self.between_segments[cluster2[0]]
                            .stats.incorrect_words_length
                            > max_intersegment_incorrect_words_length):
                        return (None, new_cluster, True)
                    merged_segment.merge_adjacent_segment(
                        self.between_segments[cluster2[0]])

            if cluster2[0] < len(self.segments):
                merged_segment.merge_adjacent_segment(
                    self.merged_segments[tuple(cluster2)])
            # else:
            # Already done
            # merged_segment.merge_adjacent_segment(self.between_segments[-1])

            self.merged_segments[new_cluster_tup] = merged_segment
            return (merged_segment, new_cluster, False)
        except:
            _global_logger.error("Failed merging cluster1 %s and cluster2 %s",
                                 cluster1, cluster2)
            for i in (cluster1 + cluster2):
                if i >= 0 and i < len(self.segments):
                    _global_logger.error("Segment %d = %s", i,
                                         self.segments[i])
            raise

    def merge_clusters(self, scoring_function,
                       max_wer=10, max_bad_proportion=0.3,
                       max_segment_length=10,
                       max_intersegment_incorrect_words_length=1):
        for i, x in enumerate(self.segments):
            _global_logger.debug("before agglomerative clustering, segment %d"
                                 " = %s", i, x)

        # Initial clusters are the individual segments themselves.
        clusters = [[x] for x in range(-1, len(self.segments) + 1)]

        rejected_clusters = set()

        while len(clusters) > 1:
            try:
                _global_logger.debug("Current clusters: %s", clusters)

                heap = []

                for i in range(len(clusters) - 1):
                    merged_segment, new_cluster, reject = (
                        self._get_merged_cluster(
                            clusters[i], clusters[i + 1], rejected_clusters,
                            max_intersegment_incorrect_words_length=(
                                max_intersegment_incorrect_words_length)))
                    if reject:
                        rejected_clusters.add(tuple(new_cluster))
                        continue
                    heapq.heappush(heap, ((-scoring_function(merged_segment), i),
                                          (merged_segment, i, new_cluster)))

                candidate_index = -1
                candidate_cluster = None

                while True:
                    try:
                        score, tup = heapq.heappop(heap)
                    except IndexError:
                        break

                    segment, index, cluster = tup

                    _global_logger.debug(
                        "Considering new cluster: (%d, %s)", index, cluster)

                    if segment.stats.wer() > max_wer:
                        _global_logger.debug(
                            "Rejecting cluster with "
                            "WER%% %.2f > %.2f", segment.stats.wer(), max_wer)
                        rejected_clusters.add(tuple(cluster))
                        continue

                    if segment.stats.bad_proportion() > max_bad_proportion:
                        _global_logger.debug(
                            "Rejecting cluster with bad-proportion "
                            "%.2f > %.2f", segment.stats.bad_proportion(),
                            max_bad_proportion)
                        rejected_clusters.add(tuple(cluster))
                        continue

                    if segment.stats.total_length > max_segment_length:
                        _global_logger.debug(
                            "Rejecting cluster with length "
                            "%.2f > %.2f", segment.stats.total_length,
                            max_segment_length)
                        rejected_clusters.add(tuple(cluster))
                        continue

                    candidate_index, candidate_cluster = tup[1:]
                    _global_logger.debug("Accepted cluster (%d, %s)",
                                         candidate_index, candidate_cluster)
                    break

                if candidate_index == -1:
                    return clusters

                new_clusters = []

                for i in range(candidate_index):
                    new_clusters.append(clusters[i])
                new_clusters.append(candidate_cluster)
                for i in range(candidate_index + 2, len(clusters)):
                    new_clusters.append(clusters[i])

                if len(new_clusters) >= len(clusters):
                    raise RuntimeError("Old: {0}; New: {1}".format(
                        clusters, new_clusters))
                clusters = new_clusters
            except Exception:
                _global_logger.error(
                    "Failed merging clusters %s", clusters)
                raise

        return clusters


def merge_segments(segments, args):
    if len(segments) == 0:
        _global_logger.debug("Got no segments at merging segments stage")
        return []

    def scoring_function(segment):
        stats = segment.stats
        try:
            return (-stats.wer() - args.silence_factor * stats.silence_length
                    - args.incorrect_words_factor
                    * stats.incorrect_words_length
                    - args.tainted_words_factor
                    * stats.num_tainted_words * 100.0 / stats.num_words)
        except ZeroDivisionError:
            return float("-inf")

    # Do agglomerative clustering on the initial segments with the score
    # for combining neighboring segments being the scoring_function on the
    # stats of the combined segment.
    merger = SegmentsMerger(segments)
    clusters = merger.merge_clusters(
        scoring_function, max_wer=args.max_wer,
        max_bad_proportion=args.max_bad_proportion,
        max_segment_length=args.max_segment_length_for_merging,
        max_intersegment_incorrect_words_length=(
            args.max_intersegment_incorrect_words_length))

    _global_logger.debug("Clusters to be merged: %s", clusters)

    # Do the actual merging based on the clusters.
    new_segments = []
    for cluster_index, cluster in enumerate(clusters):
        _global_logger.debug(
            "Merging cluster (%d, %s)", cluster_index, cluster)

        try:
            if cluster_index == 0 and len(cluster) == 1:
                assert cluster[0] == -1
                _global_logger.debug(
                    "Not adding region before the first segment")
                # skip adding the lines before the initial segment if its
                # not merged with the initial segment
                continue
            elif cluster_index == len(clusters) - 1 and len(cluster) == 1:
                _global_logger.debug(
                    "Not adding remaining end region %s",
                    cluster[0])
                assert cluster[0] == len(segments)
                # skip adding the lines after the last segment if its
                # not merged with the last segment
                break

            new_segments.append(merger.merged_segments[tuple(cluster)])
        except Exception:
            _global_logger.error("Error with cluster (%d, %s)",
                                 cluster_index, cluster)
            raise

    segments = new_segments

    for i, x in enumerate(segments):
        _global_logger.debug(
            "after agglomerative clustering: segment %d = %s", i, x)

    assert len(segments) > 0
    segment_index = 0
    # Ignore all the initial segments that have WER > max_wer
    while segment_index < len(segments):
        segment = segments[segment_index]
        if segment.stats.wer() < args.max_wer:
            break
        segment_index += 1

    if segment_index == len(segments):
        _global_logger.debug("No merged segments were below "
                             "WER%% %.2f", args.max_wer)
        return []

    _global_logger.debug("Merging overlapping segments starting from the "
                         "first segment with WER%% < max_wer i.e. %d = %s",
                         segment_index, segments[segment_index])

    new_segments = [segments[segment_index]]
    segment_index += 1
    while segment_index < len(segments):
        if segments[segment_index].stats.wer() > args.max_wer:
            # ignore this segment
            segment_index += 1
            continue
        if new_segments[-1].end_time() >= segments[segment_index].start_time():
            new_segments[-1].merge_with_segment(
                segments[segment_index], args.max_deleted_words)
        else:
            new_segments.append(segments[segment_index])
        segment_index += 1
    segments = new_segments

    return segments


def get_segments_for_utterance(split_lines_of_utt, args, utterance_stats):
    """
    This function creates the segments for an utterance as a list
    of class Segment.
    It returns a 2-tuple (list-of-segments, list-of-deleted-segments)
    where the deleted segments are only useful for diagnostic printing.
    Note: split_lines_of_utt is a list of lists, one per line, each containing
    the sequence of fields.
    """
    utterance_stats.num_utterances += 1

    segment_ranges = compute_segment_cores(split_lines_of_utt)

    utterance_end_time = (float(split_lines_of_utt[-1][2])
                          + float(split_lines_of_utt[-1][3]))
    utterance_stats.total_length_of_utterances += utterance_end_time

    segments = [Segment(split_lines_of_utt, x[0], x[1])
                for x in segment_ranges]

    utterance_stats.accumulate_segment_stats(
        segments, 'stage  0 [segment cores]')

    for i, x in enumerate(segments):
        _global_logger.debug("stage 0: segment %d = %s", i, x)

    if args.verbose > 4:
        print("Stage 0 [segment cores]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    for segment in segments:
        segment.possibly_add_tainted_lines()
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  1 [add tainted lines]')

    for i, x in enumerate(segments):
        _global_logger.debug("stage 1: segment %d = %s", i, x)

    if args.verbose > 4:
        print("Stage 1 [add tainted lines]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    segments = merge_segments(segments, args)
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  2 [merge segments]')

    for i, x in enumerate(segments):
        _global_logger.debug("stage 2: segment %d = %s", i, x)

    if args.verbose > 4:
        print("Stage 2 [merge segments]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    new_segments = []
    for s in segments:
        new_segments += s.possibly_split_segment(
            args.max_internal_silence_length,
            args.max_internal_non_scored_length)
    segments = new_segments
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  3 [split segments]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 3: segment %d, %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 3 [split segments]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    new_segments = []
    for s in segments:
        new_segments += s.possibly_split_long_segment(
            args.max_segment_length_for_splitting,
            args.hard_max_segment_length,
            args.min_silence_length_to_split,
            args.min_non_scored_length_to_split)
    segments = new_segments
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  4 [split long segments]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 4: segment %d, %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 4 [split long segments]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    for s in segments:
        s.possibly_truncate_boundaries(args.max_edge_silence_length,
                                       args.max_edge_non_scored_length)
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  5 [truncate boundaries]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 5: segment %d = %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 5 [truncate boundaries]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    for s in segments:
        s.relax_boundary_truncation(args.min_segment_length,
                                    args.min_new_segment_length)
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  6 [relax boundary truncation]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 6: segment %d = %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 6 [relax boundary truncation]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    for s in segments:
        s.possibly_add_unk_padding(args.unk_padding)
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  7 [unk-padding]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 7: segment %d = %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 7 [unk-padding]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    deleted_segments = []
    new_segments = []
    for s in segments:
        # the 0.999 allows for roundoff error.
        if (not s.is_whole_utterance()
                and s.length() < 0.999 * args.min_new_segment_length):
            s.debug_str += '[deleted-because-of--min-new-segment-length]'
            deleted_segments.append(s)
        else:
            new_segments.append(s)
    segments = new_segments
    utterance_stats.accumulate_segment_stats(
        segments,
        'stage  8 [remove new segments under --min-new-segment-length')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 8: segment %d = %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 8 [remove new segments under "
              "--min-new-segment-length]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    new_segments = []
    for s in segments:
        # the 0.999 allows for roundoff error.
        if s.length() < 0.999 * args.min_segment_length:
            s.debug_str += '[deleted-because-of--min-segment-length]'
            deleted_segments.append(s)
        else:
            new_segments.append(s)
    segments = new_segments
    utterance_stats.accumulate_segment_stats(
        segments, 'stage  9 [remove segments under --min-segment-length]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 9: segment %d = %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 9 [remove segments under "
              "--min-segment-length]:", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    new_segments = []
    for s in segments:
        if s.contains_atleast_one_scored_non_oov_word():
            new_segments.append(s)
        else:
            s.debug_str += '[deleted-because-no-scored-non-oov-words]'
            deleted_segments.append(s)
    segments = new_segments
    utterance_stats.accumulate_segment_stats(
        segments, 'stage 10 [remove segments without scored,non-OOV words]')

    for i, x in enumerate(segments):
        _global_logger.debug(
            "stage 10: segment %d = %s", i, x.debug_info(False))

    if args.verbose > 4:
        print("Stage 10 [remove segments without scored, non-OOV words "
              "", file=sys.stderr)
        segments_copy = [x.copy() for x in segments]
        print_debug_info_for_utterance(sys.stderr,
                                       copy.deepcopy(split_lines_of_utt),
                                       segments_copy, [])

    for i in range(len(segments) - 1):
        if segments[i].end_time() > segments[i + 1].start_time():
            # this just adds something to --ctm-edits-out output
            segments[i + 1].debug_str += ",overlaps-previous-segment"

    if len(segments) == 0:
        utterance_stats.num_utterances_without_segments += 1

    return (segments, deleted_segments)


def float_to_string(f):
    """ this prints a number with a certain number of digits after the point,
    while removing trailing zeros.
    """
    num_digits = 6  # we want to print 6 digits after the zero
    g = f
    while abs(g) > 1.0:
        g *= 0.1
        num_digits += 1
    format_str = '%.{0}g'.format(num_digits)
    return format_str % f


def time_to_string(time, frame_length):
    """ Gives time in string form as an exact multiple of the frame-length,
    e.g. 0.01 (after rounding).
    """
    n = round(time / frame_length)
    assert n >= 0
    # The next function call will remove trailing zeros while printing it, so
    # that e.g. 0.01 will be printed as 0.01 and not 0.0099999999999999.  It
    # seems that doing this in a simple way is not really possible (at least,
    # not without assuming that frame_length is of the form 10^-n, which we
    # don't really want to do).
    return float_to_string(n * frame_length)


def write_segments_for_utterance(text_output_handle, segments_output_handle,
                                 old_utterance_name, segments, oov_symbol,
                                 eps_symbol="<eps>", frame_length=0.01):
    num_digits = len(str(len(segments)))
    for n, segment in enumerate(segments):
        # split utterances will be named foo-bar-1 foo-bar-2, etc.
        new_utterance_name = "{old}-{index:0{width}}".format(
                                 old=old_utterance_name, index=n+1,
                                 width=num_digits)
        # print a line to the text output of the form like
        # <new-utterance-id> <text>
        # like:
        # foo-bar-1 hello this is dan
        print(new_utterance_name, segment.text(oov_symbol, eps_symbol),
              file=text_output_handle)
        # print a line to the segments output of the form
        # <new-utterance-id> <old-utterance-id> <start-time> <end-time>
        # like:
        # foo-bar-1 foo-bar 5.1 7.2
        print(new_utterance_name, old_utterance_name,
              time_to_string(segment.start_time(), frame_length),
              time_to_string(segment.end_time(), frame_length),
              file=segments_output_handle)


# Note, this is destrutive of 'segments_for_utterance', but it won't matter.
def print_debug_info_for_utterance(ctm_edits_out_handle,
                                   split_lines_of_cur_utterance,
                                   segments_for_utterance,
                                   deleted_segments_for_utterance,
                                   frame_length=0.01):
    # info_to_print will be list of 2-tuples
    # (time, 'start-segment-n'|'end-segment-n')
    # representing the start or end times of segments.
    info_to_print = []
    for n, segment in enumerate(segments_for_utterance):
        start_string = 'start-segment-{0}[{1}]'.format(n + 1,
                                                       segment.debug_info())
        info_to_print.append((segment.start_time(), start_string))
        end_string = 'end-segment-{0}'.format(n + 1)
        info_to_print.append((segment.end_time(), end_string))
    # for segments that were deleted we print info like
    # start-deleted-segment-1, and otherwise similar info to segments that were
    # retained.
    for n, segment in enumerate(deleted_segments_for_utterance):
        start_string = 'start-deleted-segment-{0}[{1}]'.format(
            n + 1, segment.debug_info(False))
        info_to_print.append((segment.start_time(), start_string))
        end_string = 'end-deleted-segment-{0}'.format(n + 1)
        info_to_print.append((segment.end_time(), end_string))

    info_to_print = sorted(info_to_print)

    for i, split_line in enumerate(split_lines_of_cur_utterance):
        # add an index like [0], [1], to the utterance-id so we can easily look
        # up segment indexes.
        split_line[0] += '[{0}]'.format(i)
        start_time = float(split_line[2])
        end_time = start_time + float(split_line[3])
        split_line_copy = list(split_line)
        while len(info_to_print) > 0 and info_to_print[0][0] <= end_time:
            (segment_start, string) = info_to_print[0]
            # shift the first element off of info_to_print.
            info_to_print = info_to_print[1:]
            # add a field like 'start-segment1[...]=3.21' to what we're about
            # to print.
            split_line_copy.append(
                '{0}={1}'.format(string,
                                 time_to_string(segment_start, frame_length)))
        print(' '.join(split_line_copy), file=ctm_edits_out_handle)


class WordStats(object):
    """
    This accumulates word-level stats about, for each reference word, with
    what probability it will end up in the core of a segment.  Words with
    low probabilities of being in segments will generally be associated
    with some kind of error (there is a higher probability of having a
    wrong lexicon entry).
    """
    def __init__(self):
        self.word_count_pair = defaultdict(lambda: [0, 0])

    def accumulate_for_utterance(self, split_lines_of_utt,
                                 segments_for_utterance,
                                 eps_symbol="<eps>"):
        # word_count_pair is a map from a string (the word) to
        # a list [total-count, count-not-within-segments]
        line_is_in_segment = [False] * len(split_lines_of_utt)
        for segment in segments_for_utterance:
            for i in range(segment.start_index, segment.end_index):
                line_is_in_segment[i] = True
        for i, split_line in enumerate(split_lines_of_utt):
            this_ref_word = split_line[6]
            if this_ref_word != eps_symbol:
                self.word_count_pair[this_ref_word][0] += 1
                if not line_is_in_segment[i]:
                    self.word_count_pair[this_ref_word][1] += 1

    def print(self, word_stats_out):
        # Sort from most to least problematic.  We want to give more prominence
        # to words that are most frequently not in segments, but also to
        # high-count words.  Define badness = pair[1] / pair[0], and
        # total_count = pair[0], where 'pair' is a value of word_count_pair.
        # We'll reverse sort on badness^3 * total_count = pair[1]^3 /
        # pair[0]^2.
        for key, pair in sorted(
                self.word_count_pair.items(),
                key=lambda item: (item[1][1] ** 3) * 1.0 / (item[1][0] ** 2),
                reverse=True):
            badness = pair[1] * 1.0 / pair[0]
            total_count = pair[0]
            print(key, badness, total_count, file=word_stats_out)
        try:
            word_stats_out.close()
        except:
            _global_logger.error("error closing file --word-stats-out=%s "
                                 "(full disk?)", word_stats_out.name)
            raise

        _global_logger.info(
            """please see the file %s for word-level
            statistics saying how frequently each word was excluded for a
            segment; format is <word> <proportion-of-time-excluded>
            <total-count>.  Particularly problematic words appear near the top
            of the file.""", word_stats_out.name)


def process_data(args, oov_symbol, utterance_stats, word_stats):
    """
    Most of what we're doing in the lines below is splitting the input lines
    and grouping them per utterance, before giving them to
    get_segments_for_utterance() and then printing the modified lines.
    """
    first_line = args.ctm_edits_in.readline()
    if first_line == '':
        sys.exit("segment_ctm_edits.py: empty input")
    split_pending_line = first_line.split()
    if len(split_pending_line) == 0:
        sys.exit("segment_ctm_edits.py: bad input line " + first_line)
    cur_utterance = split_pending_line[0]
    split_lines_of_cur_utterance = []

    while True:
        try:
            if (len(split_pending_line) == 0
                    or split_pending_line[0] != cur_utterance):
                # Read one whole utterance. Now process it.
                (segments_for_utterance,
                 deleted_segments_for_utterance) = get_segments_for_utterance(
                     split_lines_of_cur_utterance, args=args,
                     utterance_stats=utterance_stats)
                word_stats.accumulate_for_utterance(
                    split_lines_of_cur_utterance, segments_for_utterance)
                write_segments_for_utterance(
                    args.text_out, args.segments_out, cur_utterance,
                    segments_for_utterance, oov_symbol=oov_symbol,
                    frame_length=args.frame_length)
                if args.ctm_edits_out is not None:
                    print_debug_info_for_utterance(
                        args.ctm_edits_out, split_lines_of_cur_utterance,
                        segments_for_utterance, deleted_segments_for_utterance,
                        frame_length=args.frame_length)

                split_lines_of_cur_utterance = []
                if len(split_pending_line) == 0:
                    break
                else:
                    cur_utterance = split_pending_line[0]

            split_lines_of_cur_utterance.append(split_pending_line)
            next_line = args.ctm_edits_in.readline()
            split_pending_line = next_line.split()
            if len(split_pending_line) == 0:
                if next_line != '':
                    sys.exit("segment_ctm_edits.py: got an "
                             "empty or whitespace input line")
        except Exception:
            _global_logger.error(
                "Error with utterance %s", cur_utterance)
            raise


def read_non_scored_words(non_scored_words_file):
    for line in non_scored_words_file.readlines():
        parts = line.split()
        if not len(parts) == 1:
            raise RuntimeError(
                "segment_ctm_edits.py: bad line in non-scored-words "
                "file {0}: {1}".format(non_scored_words_file, line))
        _global_non_scored_words.add(parts[0])
    non_scored_words_file.close()


class UtteranceStats(object):

    def __init__(self):
        # segment_total_length and num_segments are maps from
        # 'stage' strings; see accumulate_segment_stats for details.
        self.segment_total_length = defaultdict(int)
        self.num_segments = defaultdict(int)
        # the lambda expression below is an anonymous function that takes no
        # arguments and returns the new list [0, 0].
        self.num_utterances = 0
        self.num_utterances_without_segments = 0
        self.total_length_of_utterances = 0

    def accumulate_segment_stats(self, segment_list, text):
        """
        Here, 'text' will be something that indicates the stage of processing,
        e.g. 'Stage 0: segment cores', 'Stage 1: add tainted lines', etc.
        """
        for segment in segment_list:
            self.num_segments[text] += 1
            self.segment_total_length[text] += segment.length()

    def print_segment_stats(self):
        _global_logger.info(
            """Number of utterances is %d, of which %.2f%% had no segments
            after all processing; total length of data in original utterances
            (in seconds) was %d""",
            self.num_utterances,
            (self.num_utterances_without_segments * 100.0
             / self.num_utterances),
            self.total_length_of_utterances)

        keys = sorted(self.segment_total_length.keys())
        for i, key in enumerate(keys):
            if i > 0:
                delta_percentage = '[%+.2f%%]' % (
                    (self.segment_total_length[key]
                     - self.segment_total_length[keys[i - 1]])
                    * 100.0 / self.total_length_of_utterances)
            _global_logger.info(
                'At %s, num-segments is %d, total length %.2f%% of '
                'original total %s',
                key, self.num_segments[key],
                (self.segment_total_length[key]
                 * 100.0 / self.total_length_of_utterances),
                delta_percentage if i > 0 else '')


def main():
    args = get_args()

    try:
        global _global_non_scored_words
        _global_non_scored_words = set()
        read_non_scored_words(args.non_scored_words_in)

        oov_symbol = None
        if args.oov_symbol_file is not None:
            try:
                line = args.oov_symbol_file.readline()
                assert len(line.split()) == 1
                oov_symbol = line.split()[0]
                assert args.oov_symbol_file.readline() == ''
                args.oov_symbol_file.close()
            except Exception:
                _global_logger.error("error reading file "
                                     "--oov-symbol-file=%s",
                                     args.oov_symbol_file.name)
                raise
        elif args.unk_padding != 0.0:
            raise ValueError(
                "if the --unk-padding option is nonzero (which "
                "it is by default, "
                "the --oov-symbol-file option must be supplied.")

        utterance_stats = UtteranceStats()
        word_stats = WordStats()
        process_data(args,
                     oov_symbol=oov_symbol, utterance_stats=utterance_stats,
                     word_stats=word_stats)

        try:
            args.text_out.close()
            args.segments_out.close()
            if args.ctm_edits_out is not None:
                args.ctm_edits_out.close()
        except:
            _global_logger.error("error closing one or more outputs "
                                 "(broken pipe or full disk?)")
            raise

        utterance_stats.print_segment_stats()
        if args.word_stats_out is not None:
            word_stats.print(args.word_stats_out)
        if args.ctm_edits_out is not None:
            _global_logger.info("detailed utterance-level debug information "
                                "is in %s", args.ctm_edits_out.name)
    except:
        _global_logger.error("Failed segmenting CTM edits")
        raise
    finally:
        try:
            args.text_out.close()
            args.segments_out.close()
            if args.ctm_edits_out is not None:
                args.ctm_edits_out.close()
        except:
            _global_logger.error("error closing one or more outputs "
                                 "(broken pipe or full disk?)")
            raise


if __name__ == '__main__':
    main()

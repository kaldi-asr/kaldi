#!/usr/bin/env python

# Copyright 2016   Vimal Manohar
#           2016   Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import argparse
import copy
import logging
import sys
from collections import defaultdict

"""
This script reads 'ctm-edits' file format that is produced by get_ctm_edits.py
and modified by modify_ctm_edits.py and taint_ctm_edits.py Its function is to
produce a segmentation and text from the ctm-edits input.

The ctm-edits file format that this script expects is as follows
<file-id> <channel> <start-time> <duration> <conf> <hyp-word> <ref-word> <edit> ['tainted']
[note: file-id is really utterance-id at this point].
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - '
                              '%(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


parser = argparse.ArgumentParser(
    description="This program produces segmentation and text information "
    "based on reading ctm-edits input format which is produced by "
    "steps/cleanup/internal/get_ctm_edits.py, steps/cleanup/internal/modify_ctm_edits.py and "
    "steps/cleanup/internal/taint_ctm_edits.py.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--min-segment-length", type=float, default=0.5,
                    help="Minimum allowed segment length (in seconds) for any "
                    "segment; shorter segments than this will be discarded.")
parser.add_argument("--min-new-segment-length", type=float, default=1.0,
                    help="Minimum allowed segment length (in seconds) for newly "
                    "created segments (i.e. not identical to the input utterances). "
                    "Expected to be >= --min-segment-length.")
parser.add_argument("--frame-length", type=float, default=0.01,
                    help="This only affects rounding of the output times; they will "
                    "be constrained to multiples of this value.")
parser.add_argument("--max-tainted-length", type=float, default=0.05,
                    help="Maximum allowed length of any 'tainted' line.  Note: "
                    "'tainted' lines may only appear at the boundary of a "
                    "segment")
parser.add_argument("--max-edge-silence-length", type=float, default=1.5,
                    help="Maximum allowed length of silence if it appears at the "
                    "edge of a segment (will be truncated).  This rule is "
                    "relaxed if such truncation would take a segment below "
                    "the --min-segment-length or --min-new-segment-length.")
parser.add_argument("--max-edge-non-scored-length", type=float, default=1.5,
                    help="Maximum allowed length of a non-scored word (noise, cough, etc.) "
                    "if it appears at the edge of a segment (will be truncated). "
                    "This rule is relaxed if such truncation would take a "
                    "segment below the --min-segment-length.")
parser.add_argument("--max-internal-silence-length", type=float, default=2.0,
                    help="Maximum allowed length of silence if it appears inside a segment "
                    "(will cause the segment to be split).")
parser.add_argument("--max-internal-non-scored-length", type=float, default=2.0,
                    help="Maximum allowed length of a non-scored word (noise, etc.) if "
                    "it appears inside a segment (will cause the segment to be "
                    "split).  Note: reference words which are real words but OOV "
                    "are not included in this category.")
parser.add_argument("--unk-padding", type=float, default=0.05,
                    help="Amount of padding with <unk> that we do if a segment boundary is "
                    "next to errors (ins, del, sub).  That is, we add this amount of "
                    "time to the segment and add the <unk> word to cover the acoustics. "
                    "If nonzero, the --oov-symbol-file option must be supplied.")
parser.add_argument("--max-junk-proportion", type=float, default=0.5,
                    help="Maximum proportion of the time of the segment that may "
                    "consist of potentially bad data, in which we include 'tainted' lines of "
                    "the ctm-edits input and unk-padding.")
parser.add_argument("--max-deleted-words-kept-when-merging", type=str, default=1,
                    help="When merging segments that are found to be overlapping or "
                    "adjacent after all other processing, keep in the transcript the "
                    "reference words that were deleted between the segments [if any] "
                    "as long as there were no more than this many reference words. "
                    "Setting this to zero will mean that any reference words that "
                    "were deleted between the segments we're about to reattach will "
                    "not appear in the generated transcript (so we'll match the hyp).")
parser.add_argument("--silence-factor", type=float, default=1,
                    help="""Weightage on the silence length when merging
                    segments""")
parser.add_argument("--incorrect-words-factor", type=float, default=1,
                    help="""Weightage on the incorrect_words_length when
                    merging segments""")
parser.add_argument("--tainted-or-incorrect-words-factor", type=float,
                    default=1, help="""Weightage on the WER including the
                    tainted words as incorrect words.""")
parser.add_argument("--max-wer", type=float, default=10.0,
                    help="Max WER of merged segments when merging")
parser.add_argument("--max-silence-length", type=float, default=10,
                    help="Maximum silence length allowed in merged segments")
parser.add_argument("--oov-symbol-file", type=str, default=None,
                    help="Filename of file such as data/lang/oov.txt which contains "
                    "the text form of the OOV word, normally '<unk>'.  Supplied as "
                    "a file to avoid complications with escaping.  Necessary if "
                    "the --unk-padding option has a nonzero value (which it does "
                    "by default.")
parser.add_argument("--ctm-edits-out", type=str,
                    help="Filename to output an extended version of the ctm-edits format "
                    "with segment start and end points noted.  This file is intended to be "
                    "read by humans; there are currently no scripts that will read it.")
parser.add_argument("--word-stats-out", type=str,
                    help="Filename for output of word-level stats, of the form "
                    "'<word> <bad-proportion> <total-count-in-ref>', e.g. 'hello 0.12 12408', "
                    "where the <bad-proportion> is the proportion of the time that this "
                    "reference word does not make it into a segment.  It can help reveal words "
                    "that have problematic pronunciations or are associated with "
                    "transcription errors.")


parser.add_argument("non_scored_words_in", metavar="<non-scored-words-file>",
                    help="Filename of file containing a list of non-scored words, "
                    "one per line. See steps/cleanup/internal/get_nonscored_words.py.")
parser.add_argument("ctm_edits_in", metavar="<ctm-edits-in>",
                    help="Filename of input ctm-edits file. "
                    "Use /dev/stdin for standard input.")
parser.add_argument("text_out", metavar="<text-out>",
                    help="Filename of output text file (same format as data/train/text, i.e. "
                    "<new-utterance-id> <word1> <word2> ... <wordN>")
parser.add_argument("segments_out", metavar="<segments-out>",
                    help="Filename of output segments.  This has the same format as data/train/segments, "
                    "but instead of <recording-id>, the second field is the old utterance-id, i.e "
                    "<new-utterance-id> <old-utterance-id> <start-time> <end-time>")

parser.add_argument("--verbosity", type=int, default=0,
                    help="Use higher verbosity for more debugging output")

args = parser.parse_args()

if args.verbosity > 2:
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

def IsTainted(split_line_of_utt):
    """Returns True if this line in ctm-edit is "tainted."""
    return len(split_line_of_utt) > 8 and split_line_of_utt[8] == 'tainted'


def ComputeSegmentCores(split_lines_of_utt, include_tainted=False):
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

    If include_tainted is true, then even the tainted lines are added to the
    core of the segment.
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
            if ((include_tainted or not IsTainted(split_lines_of_utt[i]))
                    and (edit_type == 'cor' or edit_type == 'sil'
                         or edit_type == 'fix')):
                line_is_in_segment_core[i] = True

    # extend each proto-segment backwards as far as we can:
    for i in reversed(range(0, num_lines - 1)):
        if line_is_in_segment_core[i + 1] and not line_is_in_segment_core[i]:
            edit_type = split_lines_of_utt[i][7]
            if ((include_tainted or not IsTainted(split_lines_of_utt[i]))
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


class SegmentStats:
    def __init__(self):
        self.num_incorrect_words = 0
        self.num_tainted_or_incorrect_words = 0
        self.incorrect_words_length = 0
        self.silence_and_junk_length = 0
        self.silence_length = 0
        self.num_words = 0
        self.total_length = 0
        self.segment_indexes = [-1]

    def wer(self):
        if self.num_words == 0:
            return float("inf")
        return float(self.num_incorrect_words) * 100.0 / self.num_words

    def Combine(self, other):
        self.num_incorrect_words += other.num_incorrect_words
        self.num_tainted_or_incorrect_words += (
            other.num_tainted_or_incorrect_words)
        self.incorrect_words_length += other.incorrect_words_length
        self.silence_and_junk_length += other.silence_and_junk_length
        self.silence_length += other.silence_length
        self.num_words += other.num_words
        self.total_length += other.total_length
        if len(other.segment_indexes) != 1 or other.segment_indexes[0] != -1:
            self.segment_indexes.extend(other.segment_indexes)


class Segment:

    def __init__(self, split_lines_of_utt, start_index, end_index,
                 debug_str=None, compute_stats=False):
        self.split_lines_of_utt = split_lines_of_utt
        # start_index is the index of the first line that appears in this
        # segment, and end_index is one past the last line.  This does not
        # include unk-padding.
        self.start_index = start_index
        self.end_index = end_index
        # If the following values are nonzero, then when we create the segment
        # we will add <unk> at the start and end of the segment [representing
        # partial words], with this amount of additional audio.
        self.start_unk_padding = 0.0
        self.end_unk_padding = 0.0

        # debug_str keeps track of the 'core' of the segment.
        if debug_str is None:
            debug_str = 'core-start={0},core-end={1}'.format(start_index,
                                                             end_index)
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

        if compute_stats:
            self.ComputeStats(-1)

    def __str__(self):
        return self.DebugInfo()

    def ComputeStats(self, segment_index):
        self.stats = SegmentStats()
        self.stats.segment_indexes = [segment_index]
        global non_scored_words
        for i in range(self.start_index, self.end_index):
            try:
                if self.split_lines_of_utt[i][7] not in ['cor', 'fix', 'sil']:
                    assert (self.split_lines_of_utt[i][6]
                            not in non_scored_words)
                    assert not IsTainted(self.split_lines_of_utt[i])
                    self.stats.num_incorrect_words += 1
                    self.stats.num_tainted_or_incorrect_words += 1
                    self.stats.incorrect_words_length += float(
                        self.split_lines_of_utt[i][3])
                if self.split_lines_of_utt[i][7] == 'sil':
                    self.stats.silence_length += float(
                        self.split_lines_of_utt[i][3])
                    self.stats.silence_and_junk_length += float(
                        self.split_lines_of_utt[i][3])
                else:
                    if self.split_lines_of_utt[i][6] not in non_scored_words:
                        self.stats.num_words += 1
                if IsTainted(self.split_lines_of_utt[i]):
                    self.stats.num_tainted_or_incorrect_words += 1
                self.stats.silence_and_junk_length += (
                    self.JunkDuration())
            except Exception:
                logger.error("Something went wrong when computing stats at "
                             "ctm line {0}".format(self.split_lines_of_utt[i]))
                raise

    def PossiblyAddTaintedLines(self):
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
        global non_scored_words
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
                adjacent_line_is_tainted = IsTainted(
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
                            and boundary_ref_word not in non_scored_words):
                        # Add the adjacent tainted line to the segment.
                        if b:
                            self.end_index += 1
                        else:
                            self.start_index -= 1

    def PossiblySplitSegment(self):
        """
        This is stage 2 of segment processing.
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
        global non_scored_words, args
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
                 this_duration > args.max_internal_silence_length)
                or (this_ref_word in non_scored_words
                    and this_duration > args.max_internal_non_scored_length)):
                # We split this segment at this index, dividing the word in two
                # [later on, in PossiblyTruncateBoundaries, it may be further
                # truncated.]
                # Note: we use 'index_to_split_at + 1' because the Segment
                # constructor takes an 'end-index' which is interpreted as one
                # past the end.
                new_segment = Segment(self.split_lines_of_utt, cur_start_index,
                                      index_to_split_at + 1, self.debug_str)
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
                                  self.end_index, self.debug_str)
            assert cur_start_is_split
            new_segment.start_keep_proportion = 0.5
            segments.append(new_segment)
        return segments

    def PossiblyTruncateBoundaries(self):
        """
        This is stage 3 of segment processing.
        It will truncate the silences and non-scored words at the segment
        boundaries if they are longer than the --max-edge-silence-length and
        --max-edge-non-scored-length respectively
        (and to the extent that this wouldn't take us below the
        --min-segment-length or --min-new-segment-length. See
        RelaxBoundaryTruncation()).

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
                    and this_duration > args.max_edge_silence_length):
                truncated_duration = args.max_edge_silence_length
            elif (this_ref_word in non_scored_words
                    and this_duration > args.max_edge_non_scored_length):
                truncated_duration = args.max_edge_non_scored_length
            if truncated_duration is not None:
                keep_proportion = truncated_duration / this_duration
                if b:
                    self.start_keep_proportion = keep_proportion
                else:
                    self.end_keep_proportion = keep_proportion

    def RelaxBoundaryTruncation(self):
        """
        This relaxes the segment-boundary truncation of
        PossiblyTruncateBoundaries(), if it would take us below
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
        length_cutoff = max(args.min_new_segment_length,
                            args.min_segment_length)
        length_with_truncation = self.Length()
        if length_with_truncation >= length_cutoff:
            return  # Nothing to do.
        orig_start_keep_proportion = self.start_keep_proportion
        orig_end_keep_proportion = self.end_keep_proportion
        if not IsTainted(self.split_lines_of_utt[self.start_index]):
            self.start_keep_proportion = 1.0
        if not IsTainted(self.split_lines_of_utt[self.end_index - 1]):
            self.end_keep_proportion = 1.0
        length_with_relaxed_boundaries = self.Length()
        if length_with_relaxed_boundaries <= length_cutoff:
            # Completely undo the truncation [to the extent allowed by the
            # presence of tainted lines at the start/end] if, even without
            # truncation, we'd be below the length cutoff.  This segment may be
            # removed later on (but it may not, if removing truncation makes us
            # identical to the input utterance, and the length is between
            # min_segment_length min_new_segment_length).
            return
        # Next, compute an interpolation constant a such that the
        # {start,end}_keep_proportion values will equal a *
        # [values-computed-by-PossiblyTruncateBoundaries()] + (1-a) * [completely-relaxed-values].
        # we're solving the equation:
        # length_cutoff = a * length_with_truncation + (1-a) * length_with_relaxed_boundaries
        # -> length_cutoff - length_with_relaxed_boundaries =
        #        a * (length_with_truncation - length_with_relaxed_boundaries)
        # -> a = (length_cutoff - length_with_relaxed_boundaries) / (length_with_truncation - length_with_relaxed_boundaries)
        a = ((length_cutoff - length_with_relaxed_boundaries)
             / (length_with_truncation - length_with_relaxed_boundaries))
        if a < 0.0 or a > 1.0:
            print("segment_ctm_edits.py: bad 'a' value = {0}".format(a),
                  file=sys.stderr)
            return
        self.start_keep_proportion = (
            a * orig_start_keep_proportion
            + (1 - a) * self.start_keep_proportion)
        self.end_keep_proportion = (
            a * orig_end_keep_proportion + (1 - a) * self.end_keep_proportion)
        if not abs(self.Length() - length_cutoff) < 0.01:
            print("segment_ctm_edits.py: possible problem relaxing boundary "
                  "truncation, length is {0} vs {1}".format(self.Length(),
                                                            length_cutoff),
                  file=sys.stderr)

    def PossiblyAddUnkPadding(self):
        """
        This is stage 4 of segment processing.
        This function may set start_unk_padding and end_unk_padding to nonzero
        values.  This is done if the current boundary words are real, scored
        words and we're not next to the beginning or end of the utterance.
        """
        # TODO(vimal): Go through this section again.
        for b in [True, False]:
            if b:
                this_index = self.start_index
            else:
                this_index = self.end_index - 1
            this_split_line = self.split_lines_of_utt[this_index]
            this_start_time = float(this_split_line[2])
            this_ref_word = this_split_line[6]
            this_edit = this_split_line[7]
            if this_edit == 'cor' and this_ref_word not in non_scored_words:
                # we can consider adding unk-padding.
                if b:   # start of utterance.
                    unk_padding = args.unk_padding
                    # close to beginning of file
                    if unk_padding > this_start_time:
                        unk_padding = this_start_time
                    # If we could add less than half of the specified
                    # unk-padding, don't add any (because when we add
                    # unk-padding we add the unknown-word symbol '<unk>', and if
                    # there isn't enough space to traverse the HMM we don't want
                    # to do it at all.
                    if unk_padding < 0.5 * args.unk_padding:
                        unk_padding = 0.0
                    self.start_unk_padding = unk_padding
                else:   # end of utterance.
                    this_end_time = this_start_time + float(this_split_line[3])
                    last_line = self.split_lines_of_utt[-1]
                    utterance_end_time = (float(last_line[2])
                                          + float(last_line[3]))
                    max_allowable_padding = utterance_end_time - this_end_time
                    assert max_allowable_padding > -0.01
                    unk_padding = args.unk_padding
                    if unk_padding > max_allowable_padding:
                        unk_padding = max_allowable_padding
                    # If we could add less than half of the specified
                    # unk-padding, don't add any (because when we add
                    # unk-padding we add the unknown-word symbol '<unk>',
                    # and if there isn't enough space to traverse the HMM we
                    # don't want to do it at all.
                    if unk_padding < 0.5 * args.unk_padding:
                        unk_padding = 0.0
                    self.end_unk_padding = unk_padding

    def MergeWithSegment(self, other, between_segment_stats=None):
        """
        This function will merge the segment in 'other' with the segment
        in 'self'.  It is only to be called when 'self' and 'other' are from
        the same utterance, 'other' is after 'self' in time order (based on
        the original segment cores), and self.EndTime() >= other.StartTime().
        Note: in this situation there will normally be deleted words
        between the two segments.  What this program does with the deleted
        words depends on '--max-deleted-words-kept-when-merging'.  If there
        were any inserted words in the transcript (less likely), this
        program will keep the reference.

        Note: --max-deleted-words-kept-when-merging can be set to a very
        large value to keep all the words.
        """
        try:
            assert (self.EndTime() + (0 if between_segment_stats is None
                                      else between_segment_stats.total_length)
                    >= other.StartTime())
            assert self.StartTime() < other.EndTime()
            assert self.split_lines_of_utt is other.split_lines_of_utt
        except AssertionError:
            logger.error("self: {0} other: {1}".format(self, other))
            raise

        if between_segment_stats is not None:
            self.stats.Combine(between_segment_stats)
        self.stats.Combine(other.stats)

        orig_self_end_index = self.end_index
        self.debug_str = "({0}/merged-with/{1})".format(self.debug_str,
                                                        other.debug_str)
        # everything that relates to the end of this segment gets copied
        # from 'other'.
        self.end_index = other.end_index
        self.end_unk_padding = other.end_unk_padding
        self.end_keep_proportion = other.end_keep_proportion
        # The next thing we have to do is to go over any lines of the ctm that
        # appear between 'self' and 'other', or are shared between both (this
        # would only happen for tainted silence or non-scored-word segments),
        # and decide what to do with them.  We'll keep the reference for any
        # substitutions or insertions (which anyway are unlikely to appear
        # in these merged segments).  Note: most of this happens in self.Text(),
        # but at this point we need to decide whether to mark any deletions
        # as 'discard-this-word'.
        first_index_of_overlap = min(orig_self_end_index - 1,
                                     other.start_index)
        last_index_of_overlap = max(orig_self_end_index - 1,
                                    other.start_index)
        num_deleted_words = 0
        for i in range(first_index_of_overlap, last_index_of_overlap + 1):
            edit_type = self.split_lines_of_utt[i][7]
            if edit_type == 'del':
                num_deleted_words += 1
        if num_deleted_words > args.max_deleted_words_kept_when_merging:
            for i in range(first_index_of_overlap, last_index_of_overlap + 1):
                if self.split_lines_of_utt[i][7] == 'del':
                    self.split_lines_of_utt[i].append('do-not-include-in-text')

    def StartTime(self):
        """Returns the start time of the utterance (within the enclosing
        utterance).
        This is before any rounding.
        """
        first_line = self.split_lines_of_utt[self.start_index]
        first_line_start = float(first_line[2])
        first_line_duration = float(first_line[3])
        first_line_end = first_line_start + first_line_duration
        return (first_line_end - self.start_unk_padding
                - (first_line_duration * self.start_keep_proportion))

    def DebugInfo(self, include_stats=True):
        """Returns some string-valued information about 'this' that is useful
        for debugging."""
        if include_stats:
            stats = ('wer={wer:.2f},'
                     'silence-length={silence_length:.2f},').format(
                        wer=self.stats.wer(),
                        silence_length=self.stats.silence_length)
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
                    start_time=self.StartTime(), end_time=self.EndTime(),
                    stats=stats, debug_str=self.debug_str))

    def EndTime(self):
        """Returns the start time of the utterance (within the enclosing
        utterance)."""
        last_line = self.split_lines_of_utt[self.end_index - 1]
        last_line_start = float(last_line[2])
        last_line_duration = float(last_line[3])
        return (last_line_start
                + (last_line_duration * self.end_keep_proportion)
                + self.end_unk_padding)

    def Length(self):
        """Returns the segment length in seconds."""
        return self.EndTime() - self.StartTime()

    def IsWholeUtterance(self):
        """returns true if this segment corresponds to the whole utterance that
        it's a part of (i.e. its start/end time are zero and the end-time of
        the last segment."""
        last_line_of_utt = self.split_lines_of_utt[-1]
        last_line_end_time = (float(last_line_of_utt[2])
                              + float(last_line_of_utt[3]))
        return (abs(self.StartTime() - 0.0) < 0.001
                and abs(self.EndTime() - last_line_end_time) < 0.001)

    def JunkProportion(self):
        """Returns the proportion of the duration of this segment that consists
        of unk-padding and tainted lines of input (will be between 0.0 and
        1.0)."""
        # Note: only the first and last lines could possibly be tainted as
        # that's how we create the segments; and if either or both are tainted
        # the utterance must contain other lines, so double-counting is not a
        # problem.
        junk_duration = self.start_unk_padding + self.end_unk_padding
        first_split_line = self.split_lines_of_utt[self.start_index]
        if IsTainted(first_split_line):
            first_duration = float(first_split_line[3])
            junk_duration += first_duration * self.start_keep_proportion
        last_split_line = self.split_lines_of_utt[self.end_index - 1]
        if IsTainted(last_split_line):
            last_duration = float(last_split_line[3])
            junk_duration += last_duration * self.end_keep_proportion
        return junk_duration / self.Length()

    def JunkDuration(self):
        """Returns duration of junk"""
        return self.JunkProportion() * self.Length()

    def PossiblyTruncateStartForJunkProportion(self):
        """
        This function will remove something from the beginning of the
        segment if it's possible to cleanly lop off a bit that contains
        more junk, as a proportion of its length, than 'args.junk_proportion'.
        Junk is defined as unk-padding and/or tainted segments.
        It considers as a potential split point, the first silence
        segment or non-tainted non-scored-word segment in the
        utterance.  See also PossiblyTruncateEndForJunkProportion.

        Note: --max-junk-proportion can be set to a high value to avoid
        removing junk.
        """
        begin_junk_duration = self.start_unk_padding
        first_split_line = self.split_lines_of_utt[self.start_index]
        if IsTainted(first_split_line):
            first_duration = float(first_split_line[3])
            begin_junk_duration += first_duration * self.start_keep_proportion
        if begin_junk_duration == 0.0:
            # nothing to do.
            return

        candidate_start_index = None
        # the following iterates over all lines internal to the utterance.
        for i in range(self.start_index + 1, self.end_index - 1):
            this_split_line = self.split_lines_of_utt[i]
            this_edit_type = this_split_line[7]
            this_ref_word = this_split_line[6]
            # We'll consider splitting on silence and on non-scored words.
            # (i.e. making the silence or non-scored word the left boundary of
            # the new utterance and discarding the piece to the left of that).
            if (this_edit_type == 'sil'
                    or (this_edit_type == 'cor'
                        and this_ref_word in non_scored_words)):
                candidate_start_index = i
                candidate_start_time = float(this_split_line[2])
                break  # Consider only the first potential truncation.
        if candidate_start_index is None:
            return  # Nothing to do as there is no place to split.
        candidate_removed_piece_duration = (candidate_start_time
                                            - self.StartTime())
        if (begin_junk_duration / candidate_removed_piece_duration
                < args.max_junk_proportion):
            # Nothing to do as the candidate piece to remove has too little
            # junk.
            return
        # OK, remove the piece.
        self.start_index = candidate_start_index
        self.start_unk_padding = 0.0
        self.start_keep_proportion = 1.0
        self.debug_str += ',truncated-start-for-junk'

    def PossiblyTruncateEndForJunkProportion(self):
        """
        This is like PossiblyTruncateStartForJunkProportion(), but
        acts on the end of the segment; see comments there.

        Note: --max-junk-proportion can be set to a high value to avoid
        removing junk.
        """
        end_junk_duration = self.end_unk_padding
        last_split_line = self.split_lines_of_utt[self.end_index - 1]
        if IsTainted(last_split_line):
            last_duration = float(last_split_line[3])
            end_junk_duration += last_duration * self.end_keep_proportion
        if end_junk_duration == 0.0:
            # nothing to do.
            return

        candidate_end_index = None
        # the following iterates over all lines internal to the utterance
        # (starting from the end).
        for i in reversed(range(self.start_index + 1, self.end_index - 1)):
            this_split_line = self.split_lines_of_utt[i]
            this_edit_type = this_split_line[7]
            this_ref_word = this_split_line[6]
            # We'll consider splitting on silence and on non-scored words.
            # (i.e. making the silence or non-scored word the right boundary of
            # the new utterance and discarding the piece to the right of that).
            if (this_edit_type == 'sil'
                    or (this_edit_type == 'cor'
                        and this_ref_word in non_scored_words)):
                # note: end-indexes are one past the last.
                candidate_end_index = i + 1
                candidate_end_time = (float(this_split_line[2])
                                      + float(this_split_line[3]))
                break  # Consider only the latest potential truncation.
        if candidate_end_index is None:
            return  # Nothing to do as there is no place to split.
        candidate_removed_piece_duration = self.EndTime() - candidate_end_time
        if (end_junk_duration / candidate_removed_piece_duration
                < args.max_junk_proportion):
            # Nothing to do as the candidate piece to remove has too little
            # junk.
            return
        # OK, remove the piece.
        self.end_index = candidate_end_index
        self.end_unk_padding = 0.0
        self.end_keep_proportion = 1.0
        self.debug_str += ',truncated-end-for-junk'

    def ContainsAtLeastOneScoredNonOovWord(self):
        """
        this will return true if there is at least one word in the utterance
        that's a scored word (not a non-scored word) and not an OOV word that's
        realized as unk.  This becomes a filter on keeping segments.
        """
        global non_scored_words
        for i in range(self.start_index, self.end_index):
            this_split_line = self.split_lines_of_utt[i]
            this_hyp_word = this_split_line[4]
            this_ref_word = this_split_line[6]
            this_edit = this_split_line[7]
            if (this_edit == 'cor' and this_ref_word not in non_scored_words
                    and this_ref_word == this_hyp_word):
                return True
        return False

    def Text(self):
        """Returns the text corresponding to this utterance, as a string."""
        global oov_symbol
        text_array = []
        if self.start_unk_padding != 0.0:
            text_array.append(oov_symbol)
        for i in range(self.start_index, self.end_index):
            this_split_line = self.split_lines_of_utt[i]
            this_ref_word = this_split_line[6]
            if (this_ref_word != '<eps>'
                    and this_split_line[-1] != 'do-not-include-in-text'):
                text_array.append(this_ref_word)
        if self.end_unk_padding != 0.0:
            text_array.append(oov_symbol)
        return ' '.join(text_array)


def AccumulateSegmentStats(segment_list, text):
    """
    Here, 'text' will be something that indicates the stage of processing,
    e.g. 'Stage 0: segment cores', 'Stage 1: add tainted lines', etc.
    """
    global segment_total_length, num_segments
    for segment in segment_list:
        num_segments[text] += 1
        segment_total_length[text] += segment.Length()


def PrintSegmentStats():
    global segment_total_length, num_segments, \
        num_utterances, num_utterances_without_segments, \
        total_length_of_utterances

    print('Number of utterances is %d, of which %.2f%% had no segments after '
          'all processing; total length of data in original utterances '
          '(in seconds) was %d' % (
              num_utterances,
              num_utterances_without_segments * 100.0 / num_utterances,
              total_length_of_utterances),
          file=sys.stderr)

    keys = sorted(segment_total_length.keys())
    for i in range(len(keys)):
        key = keys[i]
        if i > 0:
            delta_percentage = '[%+.2f%%]' % (
                (segment_total_length[key] - segment_total_length[keys[i - 1]])
                * 100.0 / total_length_of_utterances)
        print('At %s, num-segments is %d, total length %.2f%% of '
              'original total %s' % (
                  key, num_segments[key],
                  (segment_total_length[key]
                   * 100.0 / total_length_of_utterances),
                  delta_percentage if i > 0 else ''),
              file=sys.stderr)


def GetSegmentsForUtterance(split_lines_of_utt, include_tainted=True):
    """
    This function creates the segments for an utterance as a list
    of class Segment.
    It returns a 2-tuple (list-of-segments, list-of-deleted-segments)
    where the deleted segments are only useful for diagnostic printing.
    Note: split_lines_of_utt is a list of lists, one per line, each containing
    the sequence of fields.
    """
    global num_utterances, num_utterances_without_segments
    global total_length_of_utterances

    num_utterances += 1

    segment_ranges = ComputeSegmentCores(split_lines_of_utt, include_tainted)

    utterance_end_time = (float(split_lines_of_utt[-1][2])
                          + float(split_lines_of_utt[-1][3]))
    total_length_of_utterances += utterance_end_time

    segments = [Segment(split_lines_of_utt, x[0], x[1])
                for x in segment_ranges]

    AccumulateSegmentStats(segments, 'stage  0 [segment cores]')

    if not include_tainted:
        for segment in segments:
            segment.PossiblyAddTaintedLines()
        AccumulateSegmentStats(segments, 'stage  1 [add tainted lines]')
    # else tainted lines have already been included in stage 0

    new_segments = []
    for s in segments:
        new_segments += s.PossiblySplitSegment()
    segments = new_segments
    AccumulateSegmentStats(segments, 'stage  2 [split segments]')
    for s in segments:
        s.PossiblyTruncateBoundaries()
    AccumulateSegmentStats(segments, 'stage  3 [truncate boundaries]')
    for s in segments:
        s.RelaxBoundaryTruncation()
    AccumulateSegmentStats(segments, 'stage  4 [relax boundary truncation]')
    for s in segments:
        s.PossiblyAddUnkPadding()
    AccumulateSegmentStats(segments, 'stage  5 [unk-padding]')

    deleted_segments = []
    new_segments = []
    for s in segments:
        # the 0.999 allows for roundoff error.
        if (not s.IsWholeUtterance()
                and s.Length() < 0.999 * args.min_new_segment_length):
            s.debug_str += '[deleted-because-of--min-new-segment-length]'
            deleted_segments.append(s)
        else:
            new_segments.append(s)
    segments = new_segments
    AccumulateSegmentStats(
        segments,
        'stage  6 [remove new segments under --min-new-segment-length')

    new_segments = []
    for s in segments:
        # the 0.999 allows for roundoff error.
        if s.Length() < 0.999 * args.min_segment_length:
            s.debug_str += '[deleted-because-of--min-segment-length]'
            deleted_segments.append(s)
        else:
            new_segments.append(s)
    segments = new_segments
    AccumulateSegmentStats(
        segments, 'stage  7 [remove segments under --min-segment-length')

    for s in segments:
        s.PossiblyTruncateStartForJunkProportion()
    AccumulateSegmentStats(
        segments,
        'stage  8 [truncate segment-starts for --max-junk-proportion')

    for s in segments:
        s.PossiblyTruncateEndForJunkProportion()
    AccumulateSegmentStats(
        segments, 'stage  9 [truncate segment-ends for --max-junk-proportion')

    new_segments = []
    for s in segments:
        if s.ContainsAtLeastOneScoredNonOovWord():
            new_segments.append(s)
        else:
            s.debug_str += '[deleted-because-no-scored-non-oov-words]'
            deleted_segments.append(s)
    segments = new_segments

    AccumulateSegmentStats(
        segments, 'stage 10 [remove segments without scored,non-OOV words]')

    new_segments = []
    for s in segments:
        j = s.JunkProportion()
        if j <= args.max_junk_proportion:
            new_segments.append(s)
        else:
            s.debug_str += '[deleted-because-junk-proportion={0}]'.format(j)
            deleted_segments.append(s)
    segments = new_segments

    AccumulateSegmentStats(
        segments,
        'stage 11 [remove segments with junk exceeding --max-junk-proportion]')

    try:
        segments = MergeSegments(segments, args)
        AccumulateSegmentStats(
            segments, 'stage 12 [merge segments]')
    except Exception:
        logger.error("Failed merging segments")
        raise

    if len(segments) == 0:
        logger.debug("Got no segments after stage 12 [merge segments]")

    for x in segments:
        logger.debug("stage 12 [merged segments]: {0}".format(x))

    for i in range(len(segments) - 1):
        if segments[i].EndTime() > segments[i + 1].StartTime():
            # this just adds something to --ctm-edits-out output
            segments[i + 1].debug_str += ",overlaps-previous-segment"

    if len(segments) == 0:
        num_utterances_without_segments += 1

    return (segments, deleted_segments)


class SegmentsMerger(object):
    """This class contains methods for merging segments. It stores the
    appropriate statistics required for this process.

    Paramters:
        segments - Stores a list of initial segments
        num_incorrect_words - An array where the i^th index gives the number
            of incorrect words between i-1 to i. The length of this is
            len(segments) + 1, with the additional two being for the
            boundaries.
        num_tainted_or_incorrect_words - Similar to num_incorrect_words, but
            stores the number of tainted or incorrect words
        junk_length - Similar to num_incorrect_words, but stores the
            length of junk (silence, tainted words, incorrect_words, oov
            padding etc.)
        silence_length - Similar to num_incorrect_words, but stores the
            length of silence
    """
    def __init__(self, segments):
        self.segments = segments

        # Compute stats for each segment
        self.segment_stats = {}
        for i, x in enumerate(segments):
            x.ComputeStats(i)
            self.segment_stats[(x.stats.segment_indexes[0],)] = x.stats

        self.between_segment_stats = [SegmentStats()
                                      for i in range(len(segments) + 1)]

        self.split_lines_of_utt = None

        # Compute stats for between segments
        try:
            self.split_lines_of_utt = segments[0].split_lines_of_utt
            self._ComputeStats()
        except IndexError as e:
            logger.error("No input segments found!")
            raise e

    def _AccumulateStatsBetweenSegments(self, line_index, segment_index):
        if self.split_lines_of_utt[line_index][7] not in ['cor', 'fix', 'sil']:
            assert not IsTainted(self.split_lines_of_utt[line_index])
            self.between_segment_stats[segment_index].num_incorrect_words += 1
            self.between_segment_stats[
                segment_index].num_tainted_or_incorrect_words += 1
            self.between_segment_stats[
                segment_index].incorrect_words_length += float(
                    self.split_lines_of_utt[line_index][3])
        elif self.split_lines_of_utt[line_index][7] == 'sil':
            self.between_segment_stats[segment_index].silence_length += (
                float(self.splits_lines_of_utt[line_index][3]))
            self.between_segment_stats[
                segment_index].silence_and_junk_length += float(
                    self.split_lines_of_utt[line_index][3])
        elif IsTainted(self.split_lines_of_utt[line_index]):
            assert self.split_lines_of_utt[line_index][7] == 'fix'
            self.between_segment_stats[
                segment_index].num_tainted_or_incorrect_words += 1
            self.between_segment_stats[
                segment_index].silence_and_junk_length += float(
                    self.split_lines_of_utt[line_index][3])
        self.between_segment_stats[
            segment_index].total_length += float(
                self.split_lines_of_utt[line_index][3])

    def _ComputeStats(self):
        if self.split_lines_of_utt is None:
            return
        for line_index in range(0, self.segments[0].start_index):
            self._AccumulateStatsBetweenSegments(line_index, 0)

            unaccounted_length = (
                self.segments[0].StartTime()
                - self.between_segment_stats[0].total_length)
            self.between_segment_stats[0].silence_and_junk_length += (
                unaccounted_length)
            self.between_segment_stats[0].silence_length += (
                unaccounted_length)
            self.between_segment_stats[0].total_length += (
                unaccounted_length)

        for segment_index in range(1, len(self.segments)):
            for line_index in range(self.segments[segment_index - 1].end_index,
                                    self.segments[segment_index].start_index):
                self._AccumulateStatsBetweenSegments(line_index, segment_index)

            unaccounted_length = (
                self.segments[segment_index].StartTime()
                - self.segments[segment_index - 1].EndTime()
                - self.between_segment_stats[segment_index].total_length)

            self.between_segment_stats[
                segment_index].silence_and_junk_length += unaccounted_length
            self.between_segment_stats[
                segment_index].silence_length += unaccounted_length
            self.between_segment_stats[
                segment_index].total_length += unaccounted_length

        segment_index = len(self.segments) - 1
        for line_index in range(self.segments[segment_index].end_index,
                                len(self.split_lines_of_utt)):
            self._AccumulateStatsBetweenSegments(line_index, segment_index + 1)

            unaccounted_length = (
                sum([float(x) for x in self.split_lines_of_utt[-1][2:4]])
                - self.segments[segment_index].EndTime()
                - self.between_segment_stats[segment_index + 1].total_length)
            self.between_segment_stats[
                segment_index + 1].silence_and_junk_length += (
                    unaccounted_length)
            self.between_segment_stats[segment_index + 1].silence_length += (
                unaccounted_length)
            self.between_segment_stats[segment_index + 1].total_length += (
                unaccounted_length)

    def _GetStatsForMergedCluster(self, cluster1, cluster2):
        new_cluster = cluster1 + cluster2
        if tuple(new_cluster) in self.segment_stats:
            return self.segment_stats[tuple(new_cluster)], new_cluster

        if cluster1[-1] == -1:
            # Consider merging the cluster with the region before
            # the 0^th segment
            merged_stats = copy.deepcopy(
                self.between_segment_stats[cluster2[0]])
        else:
            merged_stats = copy.deepcopy(
                self.segment_stats[tuple(cluster1)])
            merged_stats.Combine(
                self.between_segment_stats[cluster2[0]])
        if cluster2[0] < len(self.segments):
            merged_stats.Combine(
                self.segment_stats[tuple(cluster2)])
        # else: merging the cluster with the region after the last
        # segment

        self.segment_stats[tuple(new_cluster)] = merged_stats
        return merged_stats, new_cluster

    def MergeClusters(self, scoring_function,
                      max_wer=100,
                      max_silence_length=10):
        if self.segments is None:
            return

        # Initial clusters are the individual segments themselves.
        clusters = [[x] for x in range(-1, len(self.segments) + 1)]

        while len(clusters) > 1:
            logger.debug("Current clusters: {0}".format(clusters))

            best_index = -1
            best_stats = None
            best_cluster = None
            for i in range(len(clusters) - 1):
                merged_stats, new_cluster = self._GetStatsForMergedCluster(
                    clusters[i], clusters[i + 1])

                if (best_stats is None
                        or (scoring_function(best_stats) <
                            scoring_function(merged_stats))):
                    best_stats = merged_stats
                    best_index = i
                    best_cluster = new_cluster

            logger.debug("New cluster: ({0}, {1})".format(best_index,
                                                          best_cluster))

            if best_stats.wer() > max_wer:
                logger.debug("Rejecting cluster with "
                             "WER% {0} > {1}".format(best_stats.wer(),
                                                     max_wer))
                break

            if best_stats.silence_length > max_silence_length:
                logger.debug("Rejecting cluster with silence-length "
                             "{0} > {1}".format(best_stats.silence_length,
                                                max_silence_length))
                break

            new_clusters = []

            for i in range(best_index):
                new_clusters.append(clusters[i])
            new_clusters.append(best_cluster)
            for i in range(best_index + 2, len(clusters)):
                new_clusters.append(clusters[i])

            if len(new_clusters) >= len(clusters):
                raise RuntimeError("Old: {0}; New: {1}".format(
                    clusters, new_clusters))
            clusters = new_clusters
        return clusters


def MergeSegments(segments, args):
    if len(segments) == 0:
        logger.debug("Got no segments after stage 11")
        return []

    def scoring_function(stats):
        return (-stats.wer() - args.silence_factor * stats.silence_length
                - args.incorrect_words_factor * stats.incorrect_words_length
                - args.tainted_or_incorrect_words_factor
                * stats.num_tainted_or_incorrect_words * 100.0
                / stats.num_words)

    # Do agglomerative clustering on the initial segments with the score
    # for combining neighboring segments being the scoring_function on the
    # stats of the combined segment.
    merger = SegmentsMerger(segments)
    clusters = merger.MergeClusters(scoring_function, args.max_wer,
                                    args.max_silence_length)

    split_lines_of_utt = segments[0].split_lines_of_utt

    # Do the actual merging based on the clusters.
    new_segments = []
    for cluster_index, cluster in enumerate(clusters):
        try:
            if cluster_index == 0 and len(cluster) == 1:
                assert cluster[0] == -1
                # skip adding the lines before the initial segment if its
                # not merged with the initial segment
                continue
            elif cluster_index == 0:
                assert cluster[0] == -1
                # Add the region before the first actual segment as a new
                # segment as this will be merged later with the first actual
                # segment.  This segment covers from ctm line 0 to the ctm line
                # before the first word in the first actual segment (i.e.
                # segments[cluster[1]])
                # Note: cluster here is of the form [-1, 0,...]
                new_segments.append(
                        Segment(split_lines_of_utt,
                                0, segments[cluster[1]].start_index,
                                compute_stats=True))
            else:
                if cluster[0] < len(segments):
                    # Add the first actual segment in the cluster
                    new_segments.append(segments[cluster[0]])
                else:
                    # No actual segments to add
                    assert len(cluster) == 1 and cluster[0] == len(segments)
                    break

            for i in range(1, len(cluster)):
                try:
                    if cluster[i] < len(segments):
                        # Merge a new segment belonging to this cluster.
                        new_segments[-1].MergeWithSegment(
                            segments[cluster[i]],
                            merger.between_segment_stats[cluster[i]])
                    else:
                        # No more actual segments to be merged, but we need to
                        # merged the region after the last actual segment.
                        # This segment covers the ctm line after the last word
                        # in the last actual segment and last ctm line ever.
                        # Note: cluster here is of the form
                        # [..., len(segments) - 1, len(segments)]
                        if (segments[cluster[i - 1]].end_index + 1
                                < len(split_lines_of_utt)):
                            new_segments[-1].MergeWithSegment(
                                Segment(split_lines_of_utt,
                                        segments[cluster[i - 1]].end_index + 1,
                                        len(split_lines_of_utt),
                                        compute_stats=True),
                                merger.between_segment_stats[-1])
                except Exception:
                    logger.error("Failed merging cluster {0} = {1} ".format(
                        i, cluster[i]))
                    logger.error("previous segment = {0}".format(
                        new_segments[-1]))
                    if cluster[i] < len(segments):
                        logger.error("next segment = {0}".format(cluster[i]))
                    else:
                        try:
                            logger.error("next segment = {0}".format(
                                Segment(split_lines_of_utt,
                                        segments[cluster[i - 1]].end_index + 1,
                                        len(split_lines_of_utt),
                                        compute_stats=True)))
                        except Exception:
                            ctm_lines = "\n".join(
                                [str(x) for x in
                                 split_lines_of_utt[
                                     segments[cluster[i - 1]].end_index + 1:]])
                            logger.error(
                                "next segment includes following "
                                "lines from ctm\n{0}".format(ctm_lines))
                            pass
                    raise
        except Exception:
            logger.error("Error with cluster {0}".format(cluster))
            raise
    segments = new_segments

    assert len(segments) > 0
    segment_index = 0
    # Ignore all the initial segments that have WER > max_wer
    while segment_index < len(segments):
        segment = segments[segment_index]
        if segment.stats.wer() < args.max_wer:
            break
        segment_index += 1

    if segment_index == len(segments):
        logger.debug("No merged segments were below "
                     "WER% {0}".format(args.max_wer))
        for x in segments:
            logger.debug("after agglomerative clustering: {0}".format(x))
        return []

    new_segments = [segment]
    while segment_index < len(segments):
        if segments[segment_index].stats.wer() > args.max_wer:
            segment_index += 1
            continue
        if new_segments[-1].EndTime() >= segments[segment_index].StartTime():
            new_segments[-1].MergeWithSegment(segments[segment_index])
        else:
            new_segments.append(segments[segment_index])
        segment_index += 1
    segments = new_segments

    return segments


def FloatToString(f):
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


def TimeToString(time, frame_length):
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
    return FloatToString(n * frame_length)


def WriteSegmentsForUtterance(text_output_handle, segments_output_handle,
                              old_utterance_name, segments):
    for n in range(len(segments)):
        segment = segments[n]
        # split utterances will be named foo-bar-1 foo-bar-2, etc.
        new_utterance_name = old_utterance_name + "-" + str(n + 1)
        # print a line to the text output of the form like
        # <new-utterance-id> <text>
        # like:
        # foo-bar-1 hello this is dan
        print(new_utterance_name, segment.Text(), file=text_output_handle)
        # print a line to the segments output of the form
        # <new-utterance-id> <old-utterance-id> <start-time> <end-time>
        # like:
        # foo-bar-1 foo-bar 5.1 7.2
        print(new_utterance_name, old_utterance_name,
              TimeToString(segment.StartTime(), args.frame_length),
              TimeToString(segment.EndTime(), args.frame_length),
              file=segments_output_handle)


# Note, this is destrutive of 'segments_for_utterance', but it won't matter.
def PrintDebugInfoForUtterance(ctm_edits_out_handle,
                               split_lines_of_cur_utterance,
                               segments_for_utterance,
                               deleted_segments_for_utterance):
    # info_to_print will be list of 2-tuples
    # (time, 'start-segment-n'|'end-segment-n')
    # representing the start or end times of segments.
    info_to_print = []
    for n in range(len(segments_for_utterance)):
        segment = segments_for_utterance[n]
        start_string = 'start-segment-' + str(n+1) + '[' + segment.DebugInfo() + ']'
        info_to_print.append((segment.StartTime(), start_string))
        end_string = 'end-segment-' + str(n + 1)
        info_to_print.append((segment.EndTime(), end_string))
    # for segments that were deleted we print info like start-deleted-segment-1, and
    # otherwise similar info to segments that were retained.
    for n in range(len(deleted_segments_for_utterance)):
        segment = deleted_segments_for_utterance[n]
        start_string = 'start-deleted-segment-' + str(n+1) + '[' + segment.DebugInfo(False) + ']'
        info_to_print.append((segment.StartTime(), start_string))
        end_string = 'end-deleted-segment-' + str(n + 1)
        info_to_print.append((segment.EndTime(), end_string))

    info_to_print = sorted(info_to_print)

    for i in range(len(split_lines_of_cur_utterance)):
        split_line = split_lines_of_cur_utterance[i]
        # add an index like [0], [1], to the utterance-id so we can easily look
        # up segment indexes.
        split_line[0] += '[' + str(i) + ']'
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
                string + "=" + TimeToString(segment_start, args.frame_length))
        print(' '.join(split_line_copy), file=ctm_edits_out_handle)


def AccWordStatsForUtterance(split_lines_of_utt,
                             segments_for_utterance):
    """
    This accumulates word-level stats about, for each reference word, with what
    probability it will end up in the core of a segment.  Words with low
    probabilities of being in segments will generally be associated with some
    kind of error (there is a higher probability of having a wrong lexicon
    entry).
    """
    # word_count_pair is a map from a string (the word) to
    # a list [total-count, count-not-within-segments]
    global word_count_pair
    line_is_in_segment = [False] * len(split_lines_of_utt)
    for segment in segments_for_utterance:
        for i in range(segment.start_index, segment.end_index):
            line_is_in_segment[i] = True
    for i in range(len(split_lines_of_utt)):
        this_ref_word = split_lines_of_utt[i][6]
        if this_ref_word != '<eps>':
            word_count_pair[this_ref_word][0] += 1
            if not line_is_in_segment[i]:
                word_count_pair[this_ref_word][1] += 1


def PrintWordStats(word_stats_out):
    try:
        f = open(word_stats_out, 'w')
    except:
        sys.exit("segment_ctm_edits.py: error opening word-stats file --word-stats-out={0} "
                 "for writing".format(word_stats_out))
    global word_count_pair
    # Sort from most to least problematic.  We want to give more prominence to
    # words that are most frequently not in segments, but also to high-count
    # words.  Define badness = pair[1] / pair[0], and total_count = pair[0],
    # where 'pair' is a value of word_count_pair.  We'll reverse sort on
    # badness^3 * total_count = pair[1]^3 / pair[0]^2.
    for key, pair in sorted(
            word_count_pair.items(),
            key=lambda item: (item[1][1] ** 3) * 1.0 / (item[1][0] ** 2),
            reverse=True):
        badness = pair[1] * 1.0 / pair[0]
        total_count = pair[0]
        print(key, badness, total_count, file=f)
    try:
        f.close()
    except:
        sys.exit("segment_ctm_edits.py: error closing file --word-stats-out={0} "
                 "(full disk?)".format(word_stats_out))
    print("segment_ctm_edits.py: please see the file {0} for word-level statistics "
          "saying how frequently each word was excluded for a segment; format is "
          "<word> <proportion-of-time-excluded> <total-count>.  Particularly "
          "problematic words appear near the top of the file.".format(
              word_stats_out),
          file=sys.stderr)


def ProcessData():
    try:
        f_in = open(args.ctm_edits_in)
    except:
        sys.exit("segment_ctm_edits.py: error opening ctm-edits input "
                 "file {0}".format(args.ctm_edits_in))
    try:
        text_output_handle = open(args.text_out, 'w')
    except:
        sys.exit("segment_ctm_edits.py: error opening text output "
                 "file {0}".format(args.text_out))
    try:
        segments_output_handle = open(args.segments_out, 'w')
    except:
        sys.exit("segment_ctm_edits.py: error opening segments output "
                 "file {0}".format(args.text_out))
    if args.ctm_edits_out is not None:
        try:
            ctm_edits_output_handle = open(args.ctm_edits_out, 'w')
        except:
            sys.exit("segment_ctm_edits.py: error opening ctm-edits output "
                     "file {0}".format(args.ctm_edits_out))

    # Most of what we're doing in the lines below is splitting the input lines
    # and grouping them per utterance, before giving them to
    # GetSegmentsForUtterance()
    # and then printing the modified lines.
    first_line = f_in.readline()
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
                 deleted_segments_for_utterance) = GetSegmentsForUtterance(
                    split_lines_of_cur_utterance)
                AccWordStatsForUtterance(split_lines_of_cur_utterance,
                                         segments_for_utterance)
                WriteSegmentsForUtterance(text_output_handle,
                                          segments_output_handle,
                                          cur_utterance, segments_for_utterance)
                if args.ctm_edits_out is not None:
                    PrintDebugInfoForUtterance(ctm_edits_output_handle,
                                               split_lines_of_cur_utterance,
                                               segments_for_utterance,
                                               deleted_segments_for_utterance)
                split_lines_of_cur_utterance = []
                if len(split_pending_line) == 0:
                    break
                else:
                    cur_utterance = split_pending_line[0]

            split_lines_of_cur_utterance.append(split_pending_line)
            next_line = f_in.readline()
            split_pending_line = next_line.split()
            if len(split_pending_line) == 0:
                if next_line != '':
                    sys.exit("segment_ctm_edits.py: got an "
                             "empty or whitespace input line")
        except Exception:
            logger.error("Error with utterance {0}".format(cur_utterance))
            raise
    try:
        text_output_handle.close()
        segments_output_handle.close()
        if args.ctm_edits_out is not None:
            ctm_edits_output_handle.close()
    except:
        sys.exit("segment_ctm_edits.py: error closing one or more outputs "
                 "(broken pipe or full disk?)")


def ReadNonScoredWords(non_scored_words_file):
    global non_scored_words
    try:
        f = open(non_scored_words_file)
    except:
        sys.exit("segment_ctm_edits.py: error opening file: "
                 "--non-scored-words=" + non_scored_words_file)
    for line in f.readlines():
        a = line.split()
        if not len(line.split()) == 1:
            sys.exit("segment_ctm_edits.py: bad line in non-scored-words "
                     "file {0}: {1}".format(non_scored_words_file, line))
        non_scored_words.add(a[0])
    f.close()


non_scored_words = set()
ReadNonScoredWords(args.non_scored_words_in)

oov_symbol = None
if args.oov_symbol_file is not None:
    try:
        with open(args.oov_symbol_file) as f:
            line = f.readline()
            assert len(line.split()) == 1
            oov_symbol = line.split()[0]
            assert f.readline() == ''
    except Exception as e:
        sys.exit("segment_ctm_edits.py: error reading file --oov-symbol-file=" +
                 args.oov_symbol_file + ", error is: " + str(e))
elif args.unk_padding != 0.0:
    sys.exit("segment_ctm_edits.py: if the --unk-padding option is nonzero (which "
             "it is by default, the --oov-symbol-file option must be supplied.")

# segment_total_length and num_segments are maps from
# 'stage' strings; see AccumulateSegmentStats for details.
segment_total_length = defaultdict(int)
num_segments = defaultdict(int)
# the lambda expression below is an anonymous function that takes no arguments
# and returns the new list [0, 0].
word_count_pair = defaultdict(lambda: [0, 0])
num_utterances = 0
num_utterances_without_segments = 0
total_length_of_utterances = 0


ProcessData()
PrintSegmentStats()
if args.word_stats_out is not None:
    PrintWordStats(args.word_stats_out)
if args.ctm_edits_out is not None:
    print("segment_ctm_edits.py: detailed utterance-level debug information "
          "is in " + args.ctm_edits_out, file=sys.stderr)

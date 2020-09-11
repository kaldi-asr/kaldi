"""Classes for representing speaker turns and interacting with RTTM files."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from intervaltree import Interval, IntervalTree

from .six import python_2_unicode_compatible
from .uem import UEM
from .utils import groupby, warn, xor

__all__ = ['merge_turns', 'trim_turns', 'Turn']


# TODO: intervaltree is pure Python and a bit of a bottleneck. Explore
#       alternatives.

@python_2_unicode_compatible
class Turn(object):
    """Speaker turn class.

    A turn represents a segment of audio attributed to a single speaker.

    Parameters
    ----------
    onset : float
        Onset of turn in seconds from beginning of recording.

    offset : float, optional
        Offset of turn in seconds from beginning of recording. If None, then
        computed from ``onset`` and ``dur``.
        (Default: None)

    dur : float, optional
        Duration of turn in seconds. If None, then computed from ``onset`` and
        ``offset``.
        (Default: None)

    speaker_id : str, optional
        Speaker id.
        (Default: None)

    file_id : str, optional
        File id.
        (Default: none)
    """
    def __init__(self, onset, offset=None, dur=None, speaker_id=None,
                 file_id=None):
        if not xor(offset is None, dur is None):
            raise ValueError('Exactly one of offset or dur must be given')
        if onset < 0:
            raise ValueError('Turn onset must be >= 0 seconds')
        if offset:
            dur = offset - onset
        if dur <= 0:
            raise ValueError('Turn duration must be > 0 seconds')
        if not offset:
            offset = onset + dur
        self.onset = onset
        self.offset = offset
        self.dur = dur
        self.speaker_id = speaker_id
        self.file_id = file_id

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.onset, self.offset, self.dur, self.file_id,
                     self.speaker_id))

    def __str__(self):
        return ('FILE: %s, SPEAKER: %s, ONSET: %f, OFFSET: %f, DUR: %f' %
                (self.file_id, self.speaker_id, self.onset, self.offset,
                 self.dur))

    def __repr__(self):
        speaker_id = ("'%s'" % self.speaker_id if self.speaker_id is not None
                      else None)
        file_id = ("'%s'" % self.file_id if self.file_id is not None
                   else None)
        return ('Turn(%f, %f, None, %s, %s)' %
                (self.onset, self.offset, speaker_id, file_id))


def merge_turns(turns):
    """Merge overlapping turns by same speaker within each file."""
    # Merge separately within each file and for each speaker.
    new_turns = []
    for (file_id, speaker_id), speaker_turns in groupby(
            turns, lambda x: (x.file_id, x.speaker_id)):
        speaker_turns = list(speaker_turns)
        speaker_it = IntervalTree.from_tuples(
            [(turn.onset, turn.offset) for turn in speaker_turns])
        n_turns_pre = len(speaker_it)
        speaker_it.merge_overlaps()
        n_turns_post = len(speaker_it)
        if n_turns_post < n_turns_pre:
            speaker_turns = []
            for intrvl in speaker_it:
                speaker_turns.append(
                    Turn(intrvl.begin, intrvl.end, speaker_id=speaker_id,
                         file_id=file_id))
            speaker_turns = sorted(
                speaker_turns, key=lambda x: (x.onset, x.offset))
            # warn('Merging overlapping speaker turns. '
            #      'FILE: %s, SPEAKER: %s' % (file_id, speaker_id))
        new_turns.extend(speaker_turns)
    return new_turns


def chop_tree(tree, onset, offset):
    """Trim Intervals so that none overlap [``onset``, ``offset``].

    Intervals contained entirely within the chopped region are removed and
    those overlapping, but not contained are trimmed back. Differs from
    ``IntervalTree.chop`` in that it keeps track of which intervals in the
    tree were affected.

    This is an inplace operation.

    Parameters
    ----------
    tree : IntervalTree
        Interval tree.

    onset : float
        Onset of chopped region.

    offset : float
        Offset of chopped region.

    Returns
    -------
    affected_intervals : set of Interval
        Intervals from ``tree`` that overlap chopped region.
    """
    overlapped_intervals = set() # Intervals overlapping chopped region.
    insertions = set() # Intervals to add.

    # Identify intervals contained entirely within [onset, offset].
    overlapped_intervals.update(tree.envelop(onset, offset))

    # Identify all other intervals overlapping [onset, offset]. These belong
    # to two classes:
    # - right overlap  --  interval.begin < onset
    # - left overlap  --  offset < interval.end
    for intrvl in tree.at(onset):
        if intrvl.begin >= onset:
            continue
        overlap_dur = intrvl.end - onset
        if not overlap_dur:
            continue
        overlapped_intervals.add(intrvl)
        insertions.add(Interval(intrvl.begin, onset, intrvl.data))
    for intrvl in tree.at(offset):
        if intrvl.end <= offset:
            continue
        overlap_dur = offset - intrvl.begin
        if not overlap_dur:
            continue
        overlapped_intervals.add(intrvl)
        insertions.add(Interval(offset, intrvl.end, intrvl.data))

    # Update tree.
    for intrvl in overlapped_intervals:
        tree.discard(intrvl)
    tree.update(insertions)

    return overlapped_intervals


MAX_SESSION_DUR = 1e6 # Maximum duration (seconds) of session. Any outlandishly
                      # high number will do.

def trim_turns(turns, uem=None, score_onset=None, score_offset=None):
    """Trim turns to scoring regions defined in UEM.

    Parameters
    ----------
    turns : list of Turn
        Speaker turns.

    uem : UEM, optional
        Un-partitioned evaluation map.
        (Default: None)

    score_onset : float, optional
        Onset of scoring region in seconds from beginning of file. Only valid
        if ``uem=None``.
        (Default: None)

    score_offset : float, optional
        Offset of scoring region in seconds from beginning of file. Only
        valid if ``uem=None``.
        (Default: None)

    Returns
    -------
    trimmed_turns : list of Turn
        Trimmed turns.
    """
    # Validate arguments.
    if uem is not None:
        if not (score_onset is None and score_offset is None):
            raise ValueError('Either uem or score_onset and score_offset must '
                             'be specified.')
    else:
        if score_onset is None or score_offset is None:
            raise ValueError('Either uem or score_onset and score_offset must '
                             'be specified.')
        if score_onset < 0:
            raise ValueError('Scoring region onset must be >= 0 seconds')
        if score_offset <= score_onset:
            raise ValueError('Scoring region duration must be > 0 seconds')

    # If no UEM provided, set each file to have same scoring region:
    # (score_onset, score_offset).
    if uem is None:
        file_ids = set([turn.file_id for turn in turns])
        uem = UEM({fid : [(score_onset, score_offset)] for fid in file_ids})

    # Trim turns to scoring regions.
    new_turns = []
    for file_id, file_turns in groupby(turns, lambda x: x.file_id):
        if file_id not in uem:
            for turn in file_turns:
                warn('Skipping turn from file not in UEM. TURN: %s' % turn)
            continue

        # Remove overlaps with no score regions.
        noscore_tree = IntervalTree.from_tuples([(0.0, MAX_SESSION_DUR)])
        for score_onset, score_offset in uem[file_id]:
            noscore_tree.chop(score_onset, score_offset)
        turns_tree = IntervalTree.from_tuples(
            (turn.onset, turn.offset, turn) for turn in file_turns)
        overlapped_turns = set() # Turns found to overlap a no score region.
        for noscore_intrvl in noscore_tree:
            overlapped_intrvls = chop_tree(
                turns_tree, noscore_intrvl.begin, noscore_intrvl.end)
            overlapped_turns.update(
                [intrvl.data for intrvl in overlapped_intrvls])

        # Convert interval tree to turns.
        for intrvl in turns_tree:
            orig_turn = intrvl.data
            new_turns.append(Turn(
                intrvl.begin, intrvl.end, speaker_id=orig_turn.speaker_id,
                file_id=orig_turn.file_id))

        # Report any overlapping turns to STDERR.
        for turn in sorted(
                overlapped_turns, key=lambda x: (x.onset, x.offset)):
            warn('Truncating turn overlapping non-scoring region. TURN: %s' %
                 turn)

    return new_turns

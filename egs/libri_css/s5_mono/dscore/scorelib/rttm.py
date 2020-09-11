"""Functions for reading/writing RTTM files."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from .turn import Turn
from .utils import format_float

__all__ = ['load_rttm', 'write_rttm', 'validate_rttm']


def _parse_rttm_line(line):
    line = line.decode('utf-8').strip()
    fields = line.split()
    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)
    file_id = fields[1]
    speaker_id = fields[7]

    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError:
        raise IOError('Turn onset not FLOAT. LINE: "%s"' % line)
    if onset < 0:
        raise IOError('Turn onset < 0 seconds. LINE: "%s"' % line)

    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError:
        raise IOError('Turn duration not FLOAT. LINE: "%s"' % line)
    if dur <= 0:
        raise IOError('Turn duration <= 0 seconds. LINE: "%s"' % line)

    return Turn(onset, dur=dur, speaker_id=speaker_id, file_id=file_id)


def load_rttm(rttmf):
    """Load speaker turns from RTTM file.

    For a description of the RTTM format, consult Appendix A of the NIST RT-09
    evaluation plan.

    Parameters
    ----------
    rttmf : str
        Path to RTTM file.

    Returns
    -------
    turns : list of Turn
        Speaker turns.

    speaker_ids : set
        Speaker ids present in ``rttmf``.

    file_ids : set
        File ids present in ``rttmf``.

    References
    ----------
    NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition
    Evaluation Plan. https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    with open(rttmf, 'rb') as f:
        turns = []
        speaker_ids = set()
        file_ids = set()
        for line in f:
            if line.startswith(b'SPKR-INFO'):
                continue
            turn = _parse_rttm_line(line)
            turns.append(turn)
            speaker_ids.add(turn.speaker_id)
            file_ids.add(turn.file_id)
    return turns, speaker_ids, file_ids


def write_rttm(rttmf, turns, n_digits=3):
    """Write speaker turns to RTTM file.

    For a description of the RTTM format, consult Appendix A of the NIST RT-09
    evaluation plan.

    Parameters
    ----------
    rttmf : str
        Path to output RTTM file.

    turns : list of Turn
        Speaker turns.

    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)

    References
    ----------
    NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition
    Evaluation Plan. https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    with open(rttmf, 'wb') as f:
        for turn in turns:
            fields = ['SPEAKER',
                      turn.file_id,
                      '1',
                      format_float(turn.onset, n_digits),
                      format_float(turn.dur, n_digits),
                      '<NA>',
                      '<NA>',
                      turn.speaker_id,
                      '<NA>',
                      '<NA>']
            line = ' '.join(fields)
            f.write(line.encode('utf-8'))
            f.write(b'\n')


def validate_rttm(rttmf):
    """Validate RTTM file.

    Parameters
    ----------
    rttmf : str
        Path to RTTM file.

    Returns
    -------
    file_ids : set of str
        File ids present in ``rttmf``.

    speaker_ids : set of str
        Speaker ids present in ``rttm``.

    error_messages : list of str
         Errors encountered in file.
    """
    with open(rttmf, 'rb') as f:
        file_ids = set()
        speaker_ids = set()
        error_messages = []
        for line in f:
            if line.startswith(b'SPKR-INFO'):
                continue
            try:
                turn = _parse_rttm_line(line)
                file_ids.add(turn.file_id)
                speaker_ids.add(turn.speaker_id)
            except IOError as e:
                error_messages.append(e.args[0])
    return file_ids, speaker_ids, error_messages

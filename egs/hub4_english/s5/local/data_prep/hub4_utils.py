# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module contains utilities for preparing the HUB4 broadcast news
evaluation corpora.
"""

import os
import re
import sys


def parse_uem_line(reco, line):
    """This method parses a 'line' from the UEM for recording 'reco'
    and returns the line converted to kaldi segments format.
    The format of UEM is
    <file-id> <channel> <start-time> <end-time>

    We force the channel to be 1 and take the file-id to be the recording-id.
    """
    line = line.strip()
    if len(line) == 0 or line[0:2] == ";;":
        return None
    parts = line.split()

    if reco is None:
        reco = parts[0]

    # The channel ID is expected to be 1.
    if parts[1] != "1":
        raise TypeError("Invalid line {0}".format(line))

    start_time = float(parts[2])
    end_time = float(parts[3])

    utt = "{0}-{1:06d}-{2:06d}".format(reco, int(start_time * 100),
                                       int(end_time * 100))
    return "{0} {1} {2} {3}".format(utt, reco, start_time, end_time)


def parse_cmu_seg_line(line, prepend_reco_to_spk=False):
    """This line parses a 'line' from the CMU automatic segmentation for
    recording.
    The CMU segmentation has the following format:
    <file> <channel> <speaker> <start-time> <end-time> <condition>

    e.g.:
    h4e_98_1 1 F0-0000     0.00    28.22 F0

    We force the channel to be 1 and take the file-id to be the recording-id.
    """
    line = line.strip()
    if len(line) == 0 or line[0:2] == ";;":
        return None
    parts = line.split()

    # Actually a file, but we assuming 1-1 mapping to recording and force
    # channel to be 1.
    reco = parts[0]

    # The channel ID is expected to be 1.
    if parts[1] != "1":
        raise TypeError("Invalid line {0}".format(line))
    spk = parts[2]

    start_time = float(parts[3])
    end_time = float(parts[4])

    if prepend_reco_to_spk:
        spk = reco + '-' + spk
        utt = "{spk}-{0:06d}-{1:06d}".format(int(start_time * 100),
                                             int(end_time * 100), spk=spk)
    else:
        utt = "{spk}-{reco}-{0:06d}-{1:06d}".format(int(start_time * 100),
                                                    int(end_time * 100),
                                                    reco=reco, spk=spk)

    segment_line = "{0} {1} {st:.3f} {end:.3f}".format(
        utt, reco, st=start_time, end=end_time)
    utt2spk_line = "{0} {1}".format(utt, spk)

    return (segment_line, utt2spk_line)


def normalize_csr_transcript(text, noise_word, spoken_noise_word):
    """Normalize broadcast news transcript for audio."""
    text = text.upper()

    # Remove long event markings
    text = re.sub(r"\[[^]/]+/\]|\[/[^]/]+\]", "", text)
    # Remove comments
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Replace alternative words with a single one (second alternative)
    text = re.sub(r"\{[^}/]+/([^}/]+)[^}]*\}", r"\1", text)
    # Remove partial word completions
    text = re.sub(r"\([^)]+\)-|-\([^)]+\)", "-", text)
    # Remove accent marks and diacritics
    text = re.sub(r"\\[3-8]", "", text)

    # Remove unclear speech markings
    text = re.sub(r"\(\(([^)]*)\)\)", r"\1", text)
    text = re.sub(r"#", "", text)   # Remove overlapped speech markings
    # Remove invented word markings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Replace speaker-made noises with <SPOKEN_NOISE>
    text = re.sub(r"\[INHALING\]|\[COUGH\]|\[THROAT_CLEARING\]|\[SIGN\]",
                  spoken_noise_word, text)
    # Replace noise with <NOISE>
    text = re.sub(r"\[[^]]+\]", noise_word, text)
    text = re.sub(r"\+([^+]+)\+", r"\1", text)

    # Remove periods after letter.
    text = re.sub(r"([A-Z])\.( |$)", r"\1 ", text)
    # Replace \. with .
    text = re.sub(r"\\.", r".", text)

    text1 = []
    for word in text.split():
        if word == spoken_noise_word or word == noise_word:
            text1.append(word)
            continue

        # Remove mispronunciation brackets
        word = re.sub(r"^@(\w+)$", r"\1", word)
        # Remove everything other than the standard ASCII symbols
        word = re.sub("[^A-Za-z0-9.' _-]", "", word)
        text1.append(word)
    return " ".join(text1)


def remove_punctuations(text):
    """Remove punctuations and some other processing for text sentence."""
    # Remove HTML new lines that are not end of sentences
    text1 = re.sub("\n", " ", text)

    # Remove some markers like double dash that are normally used to separate
    # name titles in newspapers.
    text1 = re.sub(r"(&[^;]+;|--)", " ", text1)

    # Remove quotation marks
    text1 = re.sub(r"''|``|\(|\)", " ", text1)

    # Remove everything other than the standard ASCII symbols
    text1 = re.sub("[^A-Za-z0-9.' _-]", "", text1)

    # Replace multiple .'s with single and then remove isolated '.'
    text1 = re.sub(r"\.[.]+ ", ".", text1)
    text1 = re.sub(r" \. ", " ", text1)

    # Remove isolated '-'
    text1 = re.sub(r" - ", " ", text1)

    # Replace multiple spaces with single.
    text1 = re.sub(r"[ ]+", " ", text1)

    return text1


# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module contains methods for doing text normalization of broadcast news
and similar text corpora.
"""

import re


def normalize_bn_transcript(text, noise_word, spoken_noise_word):
    """Normalize broadcast news transcript for audio."""
    text.upper()
    # Remove unclear speech markings
    text = re.sub(r"\(\(([^)]*)\)\)", r"\1", text)
    text = re.sub(r"#", "", text)   # Remove overlapped speech markings
    # Remove invented word markings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[[^]]+\]", noise_word, text)
    text = re.sub(r"\{[^}]+\}", spoken_noise_word, text)
    text = re.sub(r"\+([^+]+)\+", r"\1", text)

    text1 = []
    for word in text.split():
        # Remove mispronunciation brackets
        word = re.sub(r"^@(\w+)$", r"\1", word)
        text1.append(word)
    return " ".join(text1)


def remove_punctuations(text):
    """Remove punctuations and some other processing for text sentence."""
    text1 = re.sub("\n", " ", text)
    text1 = re.sub(r"(&[^;]+;|--)", " ", text1)
    text1 = re.sub(r"''|``|\(|\)", " ", text1)
    text1 = re.sub("[^A-Za-z0-9.' _-]", "", text1)
    text1 = re.sub(r"\. ", " ", text1)
    text1 = re.sub(r"([^0-9$-])\.([^0-9]|$)", r"\1\2", text1)
    text1 = re.sub(r" - ", " ", text1)
    text1 = re.sub(r"[ ]+", " ", text1)
    return text1

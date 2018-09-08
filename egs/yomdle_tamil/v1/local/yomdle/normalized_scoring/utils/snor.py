#!/usr/bin/env python3

""" This script defines an iterator over SNOR-formatted files.
    The iterator iterates over lines, returning tuples of form (utt, utt-id).
    snor-format:
    some text goes here (id-of-utterance)
    some other text here (id-of-next-utterance)
"""

def SnorIter(fh):
    for line in fh:
        lparen_location = line.rfind("(")
        rparen_location = line.rfind(")")

        if lparen_location > 0 and line[lparen_location-1] == " ":
            lparen_location_modifier = -1
        else:
            lparen_location_modifier = 0
        utt = line[ :lparen_location + lparen_location_modifier ]
        uttid = line[ lparen_location+1 : rparen_location ]

        yield utt, uttid

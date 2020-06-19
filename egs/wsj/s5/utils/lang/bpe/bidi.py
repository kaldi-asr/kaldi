#!/usr/bin/env python3
# Copyright   2018 Chun-Chieh Chang

# This script is largely written by Stephen Rawls
# and uses the python package https://pypi.org/project/PyICU_BiDi/
# The code leaves right to left text alone and reverses left to right text.

import icu_bidi
import io
import sys
import unicodedata
# R=strong right-to-left;  AL=strong arabic right-to-left
rtl_set =  set(chr(i) for i in range(sys.maxunicode)
               if unicodedata.bidirectional(chr(i)) in ['R','AL'])
def determine_text_direction(text):
    # Easy case first
    for char in text:
        if char in rtl_set:
            return icu_bidi.UBiDiLevel.UBIDI_RTL
    # If we made it here we did not encounter any strongly rtl char
    return icu_bidi.UBiDiLevel.UBIDI_LTR

def utf8_visual_to_logical(text):
    text_dir = determine_text_direction(text)

    bidi = icu_bidi.Bidi()
    bidi.inverse = True
    bidi.reordering_mode = icu_bidi.UBiDiReorderingMode.UBIDI_REORDER_INVERSE_LIKE_DIRECT
    bidi.reordering_options = icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_DEFAULT # icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_INSERT_MARKS

    bidi.set_para(text, text_dir, None)

    res = bidi.get_reordered(0 | icu_bidi.UBidiWriteReorderedOpt.UBIDI_DO_MIRRORING | icu_bidi.UBidiWriteReorderedOpt.UBIDI_KEEP_BASE_COMBINING)

    return res

def utf8_logical_to_visual(text):
    text_dir = determine_text_direction(text)

    bidi = icu_bidi.Bidi()

    bidi.reordering_mode = icu_bidi.UBiDiReorderingMode.UBIDI_REORDER_DEFAULT
    bidi.reordering_options = icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_DEFAULT  #icu_bidi.UBiDiReorderingOption.UBIDI_OPTION_INSERT_MARKS

    bidi.set_para(text, text_dir, None)

    res = bidi.get_reordered(0 | icu_bidi.UBidiWriteReorderedOpt.UBIDI_DO_MIRRORING | icu_bidi.UBidiWriteReorderedOpt.UBIDI_KEEP_BASE_COMBINING)

    return res


##main##
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")
for line in sys.stdin:
    line = line.strip()
    line = utf8_logical_to_visual(line)[::-1]
    sys.stdout.write(line + '\n')

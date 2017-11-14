#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import sys
import argparse
import codecs
import os
import pdb


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("olexicon", help="Output kaldi format lexicon.")
    parser.add_argument("lexicon", help="Kaldi format lexicon")
    parser.add_argument("diphthongs", help="File with diphthong mapping")
    parser.add_argument("tones", help="File with tone mapping")
    return parser.parse_args() 


def main():
    args = parse_input()

    # Load diphthong map
    dp_map = {}
    with codecs.open(args.diphthongs, "r", encoding="utf-8") as f:
        for l in f:
            phone, split_phones = l.split(None, 1)
            dp_map[phone] = split_phones.split()

    # Load tone map
    tone_map = {}
    with codecs.open(args.tones, "r", encoding="utf-8") as f:
        for l in f:
            tone, split_tones = l.split(None, 1)
            tone_map[tone] = split_tones.split()

    # Process lexicon
    lexicon_out = []
    with codecs.open(args.lexicon, "r", encoding="utf-8") as f:
        for l in f:
            word, pron = l.split(None, 1)
            new_pron = ""
            for p in pron.split():
                try:
                    p, tags = p.split("_", 1)
                    # Process tags
                    tags = tags.split("_")
                    new_tags = []
                    for t in tags:
                        try:
                            new_tags += tone_map[t]
                        except KeyError:
                            new_tags += t
                except ValueError:
                    new_tags = []
                
                # Process diphthongs
                try:
                    new_phones = dp_map[p]
                except KeyError:
                    new_phones = [p]

                # Join tags and phones
                for nph in new_phones:
                    new_pron += "_".join([nph] + new_tags) + " "
    
            lexicon_out.append((word, new_pron))            
    
    # Write new lexicon
    with codecs.open(args.olexicon, "w", encoding="utf-8") as f:
        for w, new_pron in lexicon_out:
          print(u"{}\t{}".format(w, new_pron.strip()), file=f)   

if __name__ == "__main__":
    main()

#!/usr/bin/python

from __future__ import print_function
import sys
import os
import codecs
import argparse
import unicodedata


def process_transcripts(transcripts_dir, transcripts_list):
    '''
        This looks through each transcript file, and collects the words.
        Arguments: transcripts -- file with list of babel training transcripts
    '''
    transcripts = os.path.join(transcripts_dir, transcripts_list)
    with open(transcripts, "r") as f:
        transcript_files = []
        for l in f:
            l_path = os.path.join(transcripts_dir, l.strip() + ".txt")
            transcript_files.append(l_path)

    word_list = {}
    misprons = {}
    for i_f, f in enumerate(transcript_files):
        print("\rFile ", i_f + 1, "of ", len(transcript_files), end="")
        with codecs.open(f, "r", "utf-8") as fp:
            for line in fp:
                # Don't use the lines with time markers
                if not line.startswith("["):
                    words = line.strip().split(" ")
                    for w in words:
                        if (not w.startswith("<") and not
                                w.startswith("(") and not
                                w.endswith("-") and not w.startswith("-")):
                            # Get rid of mispronunciation markings
                            if (not w.startswith("*") and not
                                    w.endswith("*") and
                                    w != "~"):
                                try:
                                    word_list[w] += 1
                                except KeyError:
                                    word_list[w] = 1
                            else:
                                w = w.replace("*", "")
                                if(w != "~"):
                                    try:
                                        misprons[w] += 1
                                    except KeyError:
                                        misprons[w] = 1
    
    word_list = sorted(word_list.items(), key=lambda x: x[0])
    misprons = sorted(misprons.items(), key=lambda x: x[0])
    print("")

    return word_list, misprons


def main():
    if len(sys.argv[1:]) == 0:
        print("Usage: ./make_word_list.py "
            "<transcripts_list> <transcripts_dir> <word_list>", file=sys.stderr)
        sys.exit(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("transcripts_list", help="Path to list of training "
                        "transcripts")
    parser.add_argument("transcripts_dir", help="Path to the training "
                        "transcripts directory")
    parser.add_argument("word_list", help="Path to the generated word list"
                        " of training words")
    parser.add_argument("--misprons", help="Path to the generated word list"
                        " of mispronounced words",
                        action="store", default=None)
    args = parser.parse_args()

    # Collect words
    words, misprons = process_transcripts(args.transcripts_dir,
                                          args.transcripts_list)

    # Create the output directory if it does not already exist
    if not os.path.exists(os.path.dirname(args.word_list)):
        os.makedirs(os.path.dirname(args.word_list))

    # Print the word list
    with codecs.open(args.word_list, "w", encoding="utf-8") as f:
        for word, count in words:
            f.write("%d %s\n" % (count, word))

    if args.misprons is not None:
        with codecs.open(args.misprons, "w", encoding="utf-8") as f:
            for word, count in misprons:
                f.write("%d %s\n" % (count, word))

if __name__ == "__main__":
    main()

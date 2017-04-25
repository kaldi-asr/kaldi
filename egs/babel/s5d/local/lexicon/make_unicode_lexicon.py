#!/usr/bin/env python
  
# Copyright 2016 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

# ============ Make unicode-based graphemic lexicon =============
#
# This script takes a list of either words or words and corresponding
# morphemes and returns a kaldi format lexicon.
# ===============================================================

# Import Statements

from __future__ import print_function
import codecs
import argparse
import unicodedata
import os
import re
import sys
import numpy as np


def main():
    args = parse_input()
    baseforms = get_word_list(args.lex_in, args.fmt)
    unicode_transcription = baseform2unicode(baseforms)
    encoded_transcription, table = encode(unicode_transcription,
                                          args.tag_percentage,
                                          log=args.log)
    write_table(table, args.lex_out)
    
    # Extract dictionary of nonspeech pronunciations
    try:
        nonspeech = {}
        with codecs.open(args.nonspeech, "r", "utf-8") as f:
            for line in f:
                line_vals = line.strip().split()
                nonspeech[line_vals[0]] = line_vals[1]
    except (IOError, TypeError):
        pass
    
    # Extract dictionary of extraspeech pronunciations (normally <hes>)
    try:
        extraspeech = {}
        with codecs.open(args.extraspeech, "r", "utf-8") as f:
            for line in f:
                line_vals = line.strip().split()
                extraspeech[line_vals[0]] = line_vals[1]     
    except (IOError, TypeError):
        pass
    
    write_lexicon(baseforms, encoded_transcription, args.lex_out,
                  nonspeech=nonspeech, extraspeech=extraspeech)


def parse_input():
    '''
        Parse commandline input.
    '''
    if len(sys.argv[1:]) == 0:
        print("Usage: ./make_unicode_lexicon.py [opts] lex_in lex_out [log]")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("lex_in", help="Path of input word list optionally "
                        "paired with a baseform. 1 word per line with the "
                        "baseform separated by a tab")
    parser.add_argument("lex_out", help="Path of output output "
                        "graphemic lexicon")
    parser.add_argument("log", nargs='?', default=None,
                        help="Directory in which the logs will be stored");
    parser.add_argument("-F", "--fmt", help="Format of input word list",
                        action="store", default="word_list")
    parser.add_argument("-T", "--tag_percentage", help="Percentage of least"
                        " frequently occurring graphemes to be tagged",
                        type=float, action="store", default=0.1)
    parser.add_argument("--nonspeech", help="File with map of nonspeech words"
                        " and pronunciations", action="store", default=None)
    parser.add_argument("--extraspeech", help="File with map of extra speech"
                        " words", action="store", default=None)
    parser.add_argument("-V", "--verbose", help="Include useful print outs",
                        action="store_true")
    args = parser.parse_args()
    return args


def _read_word_list_line(line):
    try:
        count, word = line.strip().split(None, 1)
        float(count)
        return word
    except ValueError:
        return line.strip()
     
      
def get_word_list(input_file, fmt):
    '''
        Read from input file the words and potential baseforms.

        Arguments: input_file -- path to the input word list
                   fmt -- format of input word list ["word_list", "morfessor"]
        Output:
            words -- list of tuples (word, baseform)
    '''
    with codecs.open(input_file, "r", "utf-8") as f:
        if fmt == "word_list" or fmt is None:
            words = []
            for line in f:
                w = _read_word_list_line(line)
                words.append((w, w))
                assert "." not in w, "FORMAT ERROR. Use --fmt [-F] morfessor"
        elif fmt == "morfessor":
            words = []
            for line in f:
                w, bf = line.strip().split(None, 1)
                words.append((w, bf))
        else:
            sys.exit("Error: Bad input format name")

    return words


def baseform2unicode(baseforms):
    '''
        Convert each baseform in the list, baseforms, to a parsed unicode
        description stored as a list of lists of dictionaries.
    
        unicode_transcription = [
            [{'NAME':'word1_grapheme1','FIELD1':'FIELD1_VAL',...},
            {'NAME':'word1_grapheme2','FIELD1':'FIELD1_VAL',...},...],
            [{'NAME':'word2_grapheme1,'FIELD1:'FIELD1_VAL',...},
            {},...]
            ,...,[]]

        Arguments:
            baseforms -- List of tuples (word, baseform)
                         e.g. baseforms = get_word_list()
  
        Output:
            unicode_transcription -- See above description
    '''

    # Regular expression for parsing unicode descriptions
    pattern = re.compile(
        r"(?P<LANGUAGE>[^\s]+)\s"
        r"(?P<CASE>SMALL\s|CAPITAL\s)?(?P<CHAR_TYPE>"
        "(?:SUBJOINED )?LETTER |(?:INDEPENDENT VOWEL )"
        r"|(?:VOWEL SIGN )|VOWEL |SIGN "
        r"|CHARACTER |JONGSEONG |CHOSEONG |SYMBOL |MARK |DIGIT "
        r"|SEMIVOWEL |TONE |SYLLABLE |LIGATURE |KATAKANA )"
        r"(?P<NAME>((?!WITH).)+)"
        r"(?P<TAG>WITH .+)?"
        )

    # For each graphemic baseform generate a parsed unicode description
    unicode_transcription = []
    for w, bf in baseforms:
        # Initialize empty list of words
        baseform_transcription = []
        # For each grapheme parse the unicode description
        for graph in bf:
            unicode_desc = unicodedata.name(graph)
            # Use the canonical unicode decomposition
            tags = unicodedata.normalize('NFD', graph)
            match_obj = pattern.match(unicode_desc)
      
            # Grapheme's unicode description is non-standard
            if(not match_obj):
                # Underscore, dash, hastag have special meaning
                if(graph in ("_", "-", "#")):
                    graph_dict = {
                                  'CHAR_TYPE': 'LINK',
                                  'SYMBOL': graph,
                                  'NAME': graph
                                 }
                # The grapheme is whitespace
                elif(unicode_desc in ("ZERO WIDTH SPACE",
                                      "ZERO WIDTH NON-JOINER",
                                      "ZERO WIDTH JOINER",
                                      "SPACE")):
                    # Ignore whitespace
                    continue
                else:
                    graph_dict = {'SYMBOL': graph, 'NAME': 'NOT_FOUND'}
     
            # Grapheme's unicode description is standard
            else:
                graph_dict = match_obj.groupdict()
                graph_dict["SYMBOL"] = graph
            # Add tags to dictionary (The first element of tags is actually
            # the base grapheme, so we only check all tags after the first.
            if(len(tags) > 1):
                for i, t in enumerate(tags[1:]):
                    graph_dict["TAG" + str(i)] = unicodedata.name(t)
    
            # Add grapheme unicode description dictionary to baseform list
            baseform_transcription.append(graph_dict)
        # Add baseform transcription to unicode transcription list
        unicode_transcription.append(baseform_transcription)
    return unicode_transcription


def encode(unicode_transcription, tag_percentage, log=False):
    '''
        Arguments:
            unicode_transcription -- a list of words whose graphemes are
                                   respresented as a list of dictionaries whose
                                   fields contain information about parsed
                                   unicode descriptions.
      
            tag_percentage -- percent of least frequent graphemes to tag
            log -- optional printing
              
        Outputs:
            Lexicon -- Encoded baseforms
    '''
    # Constants
    VOWELS = "AEIOU"
    SKIP = "/()"

    graphemes = []
    table = []
    encoded_transcription = []
    # Accumulate grapheme statistics over corpus at some point. For now just
    # use the lexicon word list. For estimating grapheme frequency this is
    # probably sufficient since we have many words each with many
    # graphemes. We do unfortunately have to assume that case does not matter.
    # We do not count dashes, underscores, parentheses, etc. . Just letters.
    graph_list = []
    for w in unicode_transcription:
        for graph in w:
            if graph["SYMBOL"] not in "()\/,-_#.":
                graph_list.append(graph["SYMBOL"].lower())

    graph2int = {v: k for k, v in enumerate(set(graph_list))}
    int2graph = {v: k for k, v in graph2int.items()}
    graph_list_int = [graph2int[g] for g in graph_list]
    bin_edges = range(0, len(int2graph.keys()) + 1)
    graph_counts = np.histogram(graph_list_int, bins=bin_edges)[0] / float(len(graph_list_int))
    # Set count threshold to frequency that tags the bottom 10% of graphemes
    bottom_idx = int(np.floor(tag_percentage * len(graph_counts)))
    count_thresh = sorted(graph_counts)[bottom_idx]
    graph_counts_dict = {}
    for i, count in enumerate(graph_counts):
        graph_counts_dict[int2graph[i]] = count
    
    graph_counts = graph_counts_dict
  
    # Print grapheme counts to histogram
    if log is not None:
        graph_counts_sorted = sorted(graph_counts, reverse=True,
                                     key=graph_counts.get)
        logfile = "{}/grapheme_histogram.txt".format(log)
        with codecs.open(logfile, "w", "utf-8") as fp:
            fp.write("Graphemes (Count Threshold = %.6f)\n" % count_thresh)
            for g in graph_counts_sorted:
                weight = ("-" * int(np.ceil(500.0 * graph_counts[g])) +
                          " %.6f\n" % graph_counts[g])
                fp.write("%s -" % (g) + weight)

    # Find a new baseform for each word
    for w in unicode_transcription:
        word_transcription = ""

        # Find a "pronunciation" for each grapheme in the word
        for graph in w:
            # Case 1: Check that the grapheme has a unicode description type
            # ---------------------------------------------------------------
            if("CHAR_TYPE" not in [k.strip() for k in graph.keys()]):
                if(graph["SYMBOL"] == "."):
                    graph["MAP0"] = "\t"
                    if word_transcription[-1] == " ":
                        word_transcription = word_transcription[:-1] + "\t"

                elif(graph["SYMBOL"] not in SKIP):
                    graph["MAP0"] = graph["SYMBOL"].lower()
                    word_transcription += graph["MAP0"] + " "

            # Case 2: Standard Grapheme
            # ---------------------------------------------------------------
            elif(graph["CHAR_TYPE"].strip() in
                    ("LETTER", "VOWEL", "VOWEL SIGN", "SIGN")):
                # Backoff diacritics
                base_grapheme = graph["NAME"].strip().replace(" ", "-").lower()
                graph["MAP0"] = _backoff_diacritics(graph["SYMBOL"].lower(),
                                                   base_grapheme,
                                                   graph_counts,
                                                   count_thresh)
                # Add final space
                word_transcription += graph["MAP0"] + " "
      
            # Case 3: Syllable (Assume consonant vowel pattern)
            # This is basically just here for Amharic
            # ----------------------------------------------------------------
            elif(graph["CHAR_TYPE"].strip() == "SYLLABLE"):
                # Multi-word description
                if(len(graph["NAME"].strip().split(' ')) > 1):
                    g_name = graph["NAME"].strip().replace(" ", "-").lower()
                    graph["MAP0"] = g_name
                    word_transcription += graph["MAP0"] + "\t"

                # Consonant Vowel Pattern
                else:
                    cv_pattern = (r"(?P<CONSONANT>[^%s]*)(?P<VOWEL>[%s]+)" %
                                  (VOWELS, VOWELS))
                    parsed_graph = re.match(cv_pattern, graph["NAME"])
                    if(not parsed_graph):
                        sys.exit("Syllable did not obey"
                                 "consonant-vowel pattern.")

                    graph_dict = parsed_graph.groupdict()
          
                    # Get consonant if it exists
                    if("CONSONANT" in graph_dict.keys() and
                            graph_dict["CONSONANT"]):
                        graph["MAP0"] = graph_dict["CONSONANT"].lower()
                        word_transcription += graph["MAP0"] + " "
          
                    # Get vowel if it exists
                    if("VOWEL" in graph_dict.keys() and graph_dict["VOWEL"]):
                        graph["MAP1"] = graph_dict["VOWEL"].lower()
                        word_transcription += graph["MAP1"] + "\t"

            # Case 4: Commonly occurring symbols
            # ----------------------------------------------------------------
            elif(graph["CHAR_TYPE"].strip() == "LINK"):
                # Add tab for underscores (kaldi lexicon format)
                if(graph["SYMBOL"] in ("_", "#")):
                    graph["MAP0"] = "\t"
                    if(len(word_transcription) >= 3 and
                            word_transcription[-2] == "\t"):
                        word_transcription = word_transcription[:-3] + "\t"
                    elif(len(word_transcription) >= 1):
                        word_transcription += "\t"
                    else:
                        sys.exit("Unknown rule for initial underscore")
                elif(graph["SYMBOL"] == "-"):
                    graph["MAP0"] = ""
                    continue
                else:
                    sys.exit("Unknown linking symbol found.")
                    sys.exit(1)

            # Update table of observed graphemes
            if(graph["SYMBOL"] not in graphemes):
                table.append(graph)
                graphemes.append(graph["SYMBOL"])
          
        # Append the newly transcribed word
        encoded_transcription.append(word_transcription.strip())
    return encoded_transcription, table


def _backoff_diacritics(grapheme, base_grapheme, graph_counts, count_thresh):
    '''
        Add diacritics as tags if the grapheme with diacritics occurs
        infrequently. The grapheme built by successively peeling away
        diacritics until a frequent grapheme in the lexicon is discovered.
        This grapheme is then considered a distinct unit and all peeled off
        diacritics are added as kaldi style tags

        Arguments:
            grapheme -- the raw grapheme to be processed
            base_grapheme -- the grapheme with no combining marks
                             (see unicode normalization NFD for more details)
            graph_counts -- A dictionary of all seen graphemes as keys with
                            counts as values
            count_thresh -- The frequency threshold below which diacritics
                            should be peeled away
    '''
    # Initialize variables before loop
    new_grapheme = grapheme
    removed = []
    parts = unicodedata.normalize("NFD", new_grapheme)
    # Find a backed-off (in terms of number of diacritics) grapheme with count
    # above the frequency threshold (count_thresh)
    while(len(parts) > 1 and
          (graph_counts[new_grapheme] <= count_thresh)):
        new_grapheme = unicodedata.normalize("NFC", parts[0:-1])
        tag = unicodedata.name(parts[-1]).strip().replace(" ", "").lower()
        removed.append(tag)
        parts = unicodedata.normalize("NFD", new_grapheme)

    # Collect all diactritics that will not be added as tags
    split_tags = []
    for p in parts[1:]:
        split_tag = unicodedata.name(p).strip().replace(" ", "").lower()
        split_tags.append(split_tag)

    # Append non-tag diacritics to the base grapheme
    base_grapheme = "".join([base_grapheme] + split_tags)
    # Return the tagged grapheme
    return "_".join([base_grapheme] + removed)


def write_table(table, outfile):
    '''
        Creates table of graphemes and fields of each grapheme's corresponding
        unicode description.
    
        Arguments:
            table   -- table to write
            outfile -- name of the output lexicon file
    '''
  
    # Create output table name
    outfile = os.path.splitext(outfile)[0] + "_table.txt"
    # Sort keys for convenience
    table_sorted = sorted(table, key=lambda k: k["NAME"])
    # Start writing to output
    with codecs.open(outfile, "w", "utf-8") as fo:
        # Get header names
        header_names = sorted(set().union(*[d.keys() for d in table]))
        # Write headers
        for h in header_names[:-1]:
            fo.write("%s\t" % h)
    
        fo.write("%s\n" % header_names[-1])

        # Write values if present
        for t in table_sorted:
            for h in header_names[:-1]:
                if(h in t.keys() and t[h]):
                    fo.write("%s\t" % t[h])
                else:
                    fo.write("''\t")
            if(header_names[-1] in t.keys() and t[header_names[-1]]):
                fo.write("%s\n" % t[header_names[-1]])
            else:
                fo.write("''\n")


def write_lexicon(baseforms, encoded_transcription, outfile, nonspeech=None,
                  extraspeech=None):
    '''
      Write out the encoded transcription of words

      Arguments:
          words -- list of words from a word list
          encoded_transcription  -- input encoded lexicon
          outfile -- output lexicon
    '''
    # Write Lexicon File
    with codecs.open(outfile, "w", "utf-8") as f:
        # First write the non-speech words
        try:
            for w in nonspeech.iterkeys():
                f.write("%s\t%s\n" % (w, nonspeech[w]))
        except AttributeError:
            pass
        
        # Then write extra-speech words 
        try:
            for w in extraspeech.iterkeys():
                f.write("%s\t%s\n" % (w, extraspeech[w]))
        except AttributeError:
            pass    
  
        # Then write the rest of the words
        for idx, w in enumerate(baseforms):
            # This is really just for BABEL in case <hes> is written as a word
            if(w[0].lower() == "<hes>"):
                f.write("%s\t<hes>\n" % (unicode(w[0])))
            else:
                f.write("%s\t%s\n" % (unicode(w[0]),
                                      encoded_transcription[idx]))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright  2017  Jian Wang
#            2017  Johns Hopkins University (author: Daniel Povey)
# License: Apache 2.0.

import os
import argparse
import sys

import re
tab_or_space = re.compile('[ \t]+')

parser = argparse.ArgumentParser(description="This script prepares files containing integerized text, "
                                 "for consumption by nnet3-get-egs.",
                                 epilog="E.g. " + sys.argv[0] + " --vocab-file=data/rnnlm/vocab/words.txt "
                                        "--num-splits=5 "
                                        "--data-weights-file=exp/rnnlm/data_weights.txt data/rnnlm/data "
                                        "exp/rnnlm1/split_text",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--vocab-file", type=str, default='', required=True,
                    help="The vocabulary file (used to convert symbols to integers)")
parser.add_argument("--unk-word", type=str,
                    help="The unknown word; if supplied, words out of this vocabulary "
                    "will be mapped to this word while dumping the data (if not, it is "
                    "an error).  If you supply the empty string, it is as if you did "
                    "not supply this option.")
parser.add_argument("--data-weights-file", type=str, default='', required=True,
                    help="File that specifies multiplicities and weights for each data source: "
                    "e.g. if <data_dir> contains foo.txt and bar.txt, then should have lines "
                    "like 'foo 1 0.5' and 'bar 2 1.5'.  These don't have to sum to one.")
parser.add_argument("--num-splits", type=int, required=True,
                    help="The number of pieces to split up the data into.")
parser.add_argument("text_dir",
                    help="Directory in which to look for source data, as validated by validate_text_dir.py")
parser.add_argument("split_dir",
                    help="Directory in which the split-up data will be written.  Will be created "
                    "if it does not exist.")


args = parser.parse_args()



# get the name with txt and counts file path for all data sources except dev
# return a dict with key is the name of data_source, value is txt_file_path.
def get_all_data_sources_except_dev(text_dir):
    data_sources = {}
    for f in os.listdir(text_dir):
        full_path = text_dir + "/" + f
        if f == 'dev.txt' or os.path.isdir(full_path):
            continue
        if f.endswith(".txt"):
            name = f[0:-4]
            data_sources[name] = full_path

    if data_sources == {}:
        sys.exit(sys.argv[0] + ": data directory {0} contains no .txt files "
                 "(possibly excepting dev.txt).".format(text_dir))
    return data_sources


# read the data-weights for data_sources from weights_file
# return a dict with key is name of a data source,
#                    value is a tuple (repeated_times_per_epoch, weight)
def read_data_weights(weights_file, data_sources):
    data_weights = {}
    with open(weights_file, 'r', encoding="latin-1") as f:
        for line in f:
            try:
                fields = re.split(tab_or_space, line)
                assert len(fields) == 3
                if fields[0] in data_weights:
                    raise Exception("duplicated data source({0}) specified in "
                                    "data-weights: {1}".format(fields[0], weights_file))
                data_weights[fields[0]] = (int(fields[1]), float(fields[2]))
            except Exception as e:
                sys.exit(sys.argv[0] + ": bad data-weights line: '" +
                         line.rstrip("\n") + "': " + str(e))


    for name in data_sources.keys():
        if name not in data_weights:
            sys.exit(sys.argv[0] + ": Weight for data source '{0}' not set".format(name))

    return data_weights



# This function opens the file with filename 'source_filename',
# reads from it line by line, and writes the lines round-robin
# to the filehandles in the array 'output_filehandles',
# each prepended by the weights in 'weight'.
def distribute_to_outputs(source_filename, weight, output_filehandles):
    weight_str = str(weight) + ' '
    num_outputs = len(output_filehandles)
    n = 0
    try:
        f = open(source_filename, 'r', encoding="latin-1")
    except Exception as e:
        sys.exit(sys.argv[0] + ": failed to open file {0} for reading: {1} ".format(
            source_filename, str(e)))
    for line in f:
        output_filehandle = output_filehandles[n % num_outputs]
        # the line 'line' will already have a terminating newline, so we
        # suppress the extra newline by specifying end=''.
        try:
            print(weight_str + line, end='', file=output_filehandle)
        except:
            sys.exit(sys.argv[0] + ": failed to write to temporary file (disk full?)")
        n += 1
        print
    f.close()





data_sources = get_all_data_sources_except_dev(args.text_dir)
data_weights = read_data_weights(args.data_weights_file, data_sources)

if not os.path.exists(args.split_dir + "/info"):
    os.makedirs(args.split_dir +  "/info")

# set up the 'num_splits' file, which contains an integer.
with open("{0}/info/num_splits".format(args.split_dir), 'w', encoding="latin-1") as f:
    print(args.num_splits, file=f)

# e.g. set temp_files = [ 'foo/1.tmp', 'foo/2.tmp', ..., 'foo/5.tmp' ]
# we write the text data to here, later we convert to integer
# while writing to the *.txt files.
temp_files = [ "{0}/{1}.tmp".format(args.split_dir, n) for n in range(1, args.num_splits + 1) ]

# create filehandles for writing to each of these '.tmp' output files.
temp_filehandles = []
for fname in temp_files:
    try:
        temp_filehandles.append(open(fname, 'w', encoding="latin-1"))
    except Exception as e:
        sys.exit(sys.argv[0] + ": failed to open file: " + str(e) +
                 ".. if this is a max-open-filehandles limitation, you may "
                 "need to rewrite parts of this script, but a workaround "
                 "is to use fewer splits of the data (or change your OS "
                 "ulimits)")


print(sys.argv[0] + ": distributing data to temporary files")

# this loop appends text data (prepended by data-weights), from each of
# the source .txt files, to the filehandles in 'temp_filehandles'.
for name in data_sources.keys():
    source_file = data_sources[name]
    multiplicity = data_weights[name][0]
    weight = data_weights[name][1]
    assert multiplicity >= 0

    for n in range(multiplicity):
        # 'offset' will be zero for the first copy of any data, and
        # from there it will increase up to some value less than
        # args.num_splits.  The point of this offset, which you can
        # think of as a rotation modulo args.num_splits, is so that
        # when we write the same data multiple times, we don't end
        # up writing the same lines to the same file.
        offset = (n * args.num_splits) // multiplicity
        assert offset < args.num_splits
        rotated_filehandles = temp_filehandles[offset:] + temp_filehandles[:offset]

        # The following line is the core of what we're doing; see the
        # documentation for this function for more details.
        distribute_to_outputs(source_file, weight, rotated_filehandles)


for f in temp_filehandles:
    try:
        f.close()
    except:
        sys.exit(sys.argv[0] + ": error closing temporary file (disk full?)");


print(sys.argv[0] + ": converting from text to integer form.")

if args.unk_word != None and args.unk_word != '':
    unk_opt = "--map-oov '{0}'".format(args.unk_word)
else:
    unk_opt = ""

# Convert from text to integer form using the vocabulary file,
# moving data from *.tmp to *.txt.
for n in range(1, args.num_splits + 1):
    command = "utils/sym2int.pl {unk_opt} -f 2- {vocab_file} <{input_file} >{output_file}".format(
        vocab_file=args.vocab_file,
        unk_opt=unk_opt,
        input_file="{0}/{1}.tmp".format(args.split_dir, n),
        output_file="{0}/{1}.txt".format(args.split_dir, n))
    ret = os.system(command)
    if ret != 0:
        sys.exit(sys.argv[0] + ": command '{0}' returned with status {1}".format(
            command, ret))
    os.remove("{0}/{1}.tmp".format(args.split_dir, n))


print(sys.argv[0] + ": converting dev data from text to integer form.")


command = "utils/sym2int.pl {unk_opt} {vocab_file} <{input_file} | {awk_command} >{output_file}".format(
    vocab_file=args.vocab_file,
    unk_opt=unk_opt,
    awk_command="awk '{print 1.0, $0;}'",  # this is passed as a variable
                                           # because it has {}'s awhich would
                                           # otherwise be interpreted.
    input_file="{0}/dev.txt".format(args.text_dir),
    output_file="{0}/dev.txt".format(args.split_dir))
ret = os.system(command)
if ret != 0:
    sys.exit(sys.argv[0] + ": command '{0}' returned with status {1}".format(
            command, ret))


print(sys.argv[0] + ": created split data in {0}".format(args.split_dir))

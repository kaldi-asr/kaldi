#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (author: Hang Lyu)

"""This script is suitable for the training set whose size is not very big, such
as TS learning, age or emotion id.

For generating egs for xvector in a more effecient way, we use "ranges in 
script-file lines" method(for taking sub-parts of matrices) in this script 
rather than generating range.* files ever as before.

We generate new feats.scp in data_li directory. The data_li directory means
the length of all the utterances in this directory is "li".
(To describle clearly, we denote the feats.scp in data_li directory as feats_li.scp)

The number of total frames is frames(feat.scp) * num_repeat.
So frmaes(feats_li.scp) == (1/kinds_of_length) * frames(feats.scp) * num_repeat

We caculate how many segment should be generated for each utterance. And then
we average the utterance and randomly select the startpoint in each segment.
So more than 50% frames in each utterance will be used for each new directory
in theory. If the kinds_of_length > 2, all the frames in original directory
should be covered.

When we generate valid set or train_sub set, we can set the option 
"generate-validate" to true. In this way, we will only generate one eg for each
utterance. Most of time, we recommond to use geomatric distance for different 
kinds of length.

"""

from __future__ import print_function
import re, os, argparse, sys, math, warnings, random

def get_args():
    parser = argparse.ArgumentParser(description="In previous xvector setup, "
            "the different examples in the same archive are equivalent length."
            "Now, we hope the length of different examples in the same archive is "
            "different. This script modifies the feature.scp in original data directory "
            "to generate some new feats.scp which corresponds "
            "to a specific length. After then, we can combine the new files, "
            "shuffle the list and split it into pieces. At last, we use "
            "nnet3-xvector-get-egs-seg to dump egs. The format of new feats_li.scp is "
            "<original_uttid-startpoint-len> <extend-filename-of-features[startpoint, startpoint+len-1]>.",
            epilog="Called by sid/nnet3/xvector/get_egs.sh")
    parser.add_argument("--num-repeats",type=int, default=10, help="Number of "
            "times each speaker repeats within an archive.")
    parser.add_argument("--min-frames-per-chunk", type=int, default=50,
            help="Minimum number of frames-per-chunk used for any archive.")
    parser.add_argument("--max-frames-per-chunk", type=int, default=300,
            help="Maximum number of frames-per-chunk used for any archive")
    parser.add_argument("--kinds-of-length", type=int, default=4,
            help="Number of length types.")
    parser.add_argument("--generate-validate", type=str,
            help="If true, for each utterance-id, we just generate one segment."
            "This method will be used in generating validate egs.",
            default="false", choices = ["false", "true"])
    parser.add_argument("--randomize-chunk-length", type=str,
            help="If true, randomly pick a chunk length in [min-frames-per-chunk, max-frames-per-chunk]."
            "If false, the chunk length varies from min-frames-per-chunk to "
            "max-frames-per-chunk according to a geometric sequence.",
            default="true", choices = ["false", "true"])
    parser.add_argument("--seed", type=int, default=123,
            help="Seed for random number generator.")
    parser.add_argument("--data-dir", type=str, required=True,
            help="The location of original data directory which contains "
            "feats.scp ,utt2num_frames and utt2spk.")
    parser.add_argument("--output-dir", type=str, required=True,
            help="The name of output-dir. In it, there are 'kinds-of-length' directories. "
            "In each subdirectory, it will contains the new generated feats_li.scp and new utt2spk.")

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    args = process_args(args)
    return args


def process_args(args):
    if not args.generate_valiate and args.num_repeats < 1:
        raise Exception("--num-repeats should have a minimum value of 1")
    if not os.path.exists(args.data_dir):
        raise Exception("This script expects --data-dir to exist")
    if not os.path.exists(args.data_dir+"/utt2num_frames"):
        raise Exception("This script expects the utt2num_frames file to exist")
    if not os.path.exists(args.data_dir+"/feats.scp"):
        raise Exception("This script expects the original feats.scp to exist")
    if not os.path.exists(args.data_dir+"/utt2spk"):
        raise Exception("This script expects the original utt2spk to exist")
    if args.min_frames_per_chunk <= 1:
        raise Exception("--min-frames-per-chunk is invalid")
    if args.max_frames_per_chunk < args.min_frames_per_chunk:
        raise Exception("--max-frames-per-chunk is invalid")
    if args.kinds_of_length < 1:
        raise Exception("--kinds-of-length is invalid")
    return args


def get_utt2spk(utt2spk_filename):
    """This function collect utt2spk and spk2utt information from a utt2spk file
    Example usage:
    utt2spk, spk2utt = get_utt2spk(utt2spk_filename)
    """
    utt2spk = {}
    spk2utt = {}
    with open(utt2spk_filename, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) != 2:
                raise Exception("Bad line in utt2spk file {0}".format(line))
            utt = tokens[0]
            spk = tokens[1]
            utt2spk[utt] = spk
            if spk not in spk2utt:
                spk2utt[spk] = [utt]
            else:
                spk2utt[spk].append(utt)
    return utt2spk, spk2utt


def get_utt2len(utt2len_filename):
    """ The function reads a utt2len file so that it generae utt2len list.
    Example usage:
    utt2len = get_utt2len(utt2len_filename)
    """
    utt2len = {}
    with open(utt2len_filename, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) != 2:
                raise Exception("Bad line in utt2len file {0}".format(line))
            utt2len[tokens[0]] = int(tokens[1])
    return utt2len


def get_utt2feat(utt2feat_filename):
    """The function generate a utt_id list and a dict mapping utt_id to 
    corresponding feature_extension from a feats.scp file.
    Example usage:
    utt_ids, utt2feat = get_utt2feat(utt2feat_filename)
    """
    utt2feat = {}
    with open(utt2feat_filename, "r") as f:
        utt_ids = []
        for line in f:
            tokens = line.split()
            if len(tokens) != 2:
                raise Exception("Bad line in utt2len file {0}".format(line))
            utt2feat[tokens[0]] = tokens[1]
            utt_ids.append(tokens[0])
    return utt_ids, utt2feat


def random_chunk_length(min_frames_per_chunk, max_frames_per_chunk, kinds_of_len):
    """The function randomly generate N numbers which represent N kinds of 
    length without repetition.
    """
    ans = []
    while(len(ans) < kinds_of_len):
      x=random.randint(min_frames_per_chunk, max_frames_per_chunk)
      if x not in ans:
          ans.append(x)
    return ans


def deterministic_chunk_length(min_frames_per_chunk, max_frames_per_chunk, kinds_of_len):
    """This function returns a geometric sequence in the range
    min-frames-per-chunk, max-frames-per-chunk]. For example, suppose min-frames-per-chunk
    is 50, and max-frames-per-chunk is 200, and kinds-of-len is 3. The output will
    be [50, 100, 200]
    """
    ans = []
    if min_frames_per_chunk == max_frames_per_chunk:
        ans = [ max_frames_per_chunk ] * kinds_of_len
    elif kinds_of_len == 1:
        ans = [ max_frames_per_chunk ]
    else:
        for i in range(0, kinds_of_len):
            ans.append(
                int(math.pow(float(max_frames_per_chunk)/min_frames_per_chunk,
                    float(i)/(kinds_of_len-1)) * min_frames_per_chunk + 0.5))
    return ans


def create_newdir(dir_name, length):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    feats_filename = dir_name + "/feats.scp"
    try:
        feats_f = open(feats_filename, "w")
    except Exception as e:
        sys.exit("{0}: error opening file '{1}'. Error was {2}".format(
            sys.argv[0], feats_filename, repr(e)))
        
    utt2spk_filename = dir_name + "/utt2spk"
    try:
        spk_f = open(utt2spk_filename, "w")
    except Exception as e:
        sys.exit("{0}: error opening file '{1}'. Error was {2}".format(
            sys.argv[0], utt2spk_filename, repr(e)))
    return feats_f, spk_f


def print_newline(utt_id, start_point, length, utt_extend, spk_id, feats_f, spk_f):
    new_utt_id = "{0}-{1}-{2}".format(utt_id, start_point, length)
    new_extend = "{0}[{1}:{2}]".format(utt_extend, start_point, start_point+length-1)
    print("{0} {1}".format(new_utt_id, new_extend), file=feats_f)
    print("{0} {1}".format(new_utt_id, spk_id), file=spk_f)


def main():
    args = get_args()
    random.seed(args.seed)
    #utt2len is a dict: key=uttid, value=num_of_frames
    utt2len_filename = "{0}/utt2num_frames".format(args.data_dir)
    utt2len = get_utt2len(utt2len_filename)
    #utt2feat is a dict: key=uttid, value=extend_filename_of_featue
    #utt_list is a list cotains all the original uttid
    utt2feat_filename = "{0}/feats.scp".format(args.data_dir)
    utt_list, utt2feat = get_utt2feat(utt2feat_filename)
    #utt2spk is a dict: key=uttid, value=spkid
    utt2spk_filename = "{0}/utt2spk".format(args.data_dir)
    utt2spk, spk2utt = get_utt2spk(utt2spk_filename)
    spks = spk2utt.keys()

    length_types = []
    # Generate l1 ... ln
    if args.randomize_chunk_length == "true":
        length_types = random_chunk_length(args.min_frames_per_chunk, 
                args.max_frames_per_chunk, args.kinds_of_length)
    else:
        length_types = deterministic_chunk_length(args.min_frames_per_chunk,
                args.max_frames_per_chunk, args.kinds_of_length)
    

    if (args.generate_validate == "false"):
        # Generate new feats_li.scp in data_li directory
        # As we hope the total_frames == frames(feats.scp) * num_repeat
        # So frmaes(feats_li.scp) == (1/kinds_of_length) * frames(feats.scp) * num_repeat
        # So for each utterance in feats.scp, the number of it in new feats_li.scp
        # is max{(|utt|/|li|) * num_repeat / kinds_of_length, 1}
        for i in range(0, args.kinds_of_length):
            num_err = 0
            this_length = length_types[i]
            this_dir_name =  args.output_dir + "/data_" + str(this_length)
            feats_f, spk_f = create_newdir(this_dir_name, this_length)
        
            for this_utt_id in utt2spk:
                this_spk_id = utt2spk[this_utt_id]
                this_extend = utt2feat[this_utt_id]
                this_utt_len = utt2len[this_utt_id]
                if this_utt_len < this_length:
                    num_err = num_err + 1
                    if num_err > (0.1 * len(utt_list)):
                        raise Exception("{0} is not a suitable length".format(this_length))
                    continue
                else:
                    num_segs = max(int(float(this_utt_len) / this_length * 
                            args.num_repeats / args.kinds_of_length), 1)
                    this_utt_boundary = int(this_utt_len - this_length)
                    num_segs = min(num_segs, this_utt_boundary)
                    # we divid the range [0, this_utt_boundary) into num_segs portions
                    this_seg = int(this_utt_boundary / num_segs)
                    for k in range(num_segs):
                        start_point = random.randint(k*this_seg, (k+1)*this_seg )
                        print_newline(this_utt_id, start_point, this_length,
                                      this_extend, this_spk_id, feats_f, spk_f)
            feats_f.close()
            spk_f.close()
    else:
        #For each utterance, generate one segment. It always be used in validate or diagnose.
        for i in range(0, args.kinds_of_length):
            num_err = 0
            this_length = length_types[i]

            this_dir_name =  args.output_dir + "/data_" + str(this_length)
            feats_f, spk_f = create_newdir(this_dir_name, this_length)

            for this_utt_id in utt2spk:
                this_spk_id = utt2spk[this_utt_id]
                this_extend = utt2feat[this_utt_id]
                this_utt_len = utt2len[this_utt_id]
                if this_utt_len < this_length:
                    num_err = num_err + 1
                    if num_err > (0.1 * len(utt_list)):
                        raise Exception("{0} is not a suitable length".format(
                            this_length))
                    continue
                else:
                    this_utt_boundary = int(this_utt_len - this_length)
                    start_point = random.randint(0, this_utt_boundary)
                    print_newline(this_utt_id, start_point, this_length,
                            this_extend, this_spk_id, feats_f, spk_f)
            feats_f.close()
            spk_f.close()
    print("allocate_egs_seg.py: finished")



if __name__ == "__main__":
    main()


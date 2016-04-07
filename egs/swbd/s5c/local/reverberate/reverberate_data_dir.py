#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate reverberated data

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, glob, math, os, random, sys, warnings, copy, imp, ast

nodes = imp.load_source('', 'steps/nnet3/components.py')
train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

class list_cyclic_iterator:
  def __init__(self, list, random_seed = 0):
    self.list_index = 0
    self.list = list
    random.seed(random_seed)
    random.shuffle(self.list)

  def next(self):
    item = self.list[self.list_index]
    self.list_index = (self.list_index + 1) % len(self.list)
    return item

def return_nonempty_lines(lines):
  new_lines = []
  for line in lines:
    if len(line.strip()) > 0:
      new_lines.append(line.strip())

  return new_lines


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for TDNNs creation and training",
                                     epilog="See steps/nnet3/tdnn/train.sh for example.")

    parser.add_argument("--rir-list", type=str, default = None,
                        help="RIR information file")
    parser.add_argument("--noise-list", type=str, default = None,
                        help="Noise information file")
    parser.add_argument("--num-replications", type=int, dest = "num_replica", default = 1,
                        help="Number of replicate to generated for the data")
    parser.add_argument("input_dir",
                        help="Input data directory")
    parser.add_argument("output_dir",
                        help="Output data directory")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## Check arguments.
    if args.rir_list is None:
        raise Exception("Rir information file must be provided")
    
    if not os.path.isfile(args.rir_list):
        raise Exception(args.rir_list + "not found")
    
    if args.noise_list is not None:
        if not os.path.isfile(args.noise_list):
            raise Exception(args.noise_list + "not found")

    return args


def Corrupt_wav(input_dir, output_dir, rir_list, num_replica, prefix):
    rirs = list_cyclic_iterator(return_nonempty_lines(open(rir_list).readlines()), random_seed = 1)
    wav_files = open(input_dir + "/wav.scp", 'r').readlines()
    command_list = []
    for i in range(num_replica):
        for j in range(len(wav_files)):
            split1 = wav_files[j].split()
            wav_pipe = " ".join(split1[1:])
            output_wav_id = prefix + str(i) + "_" + split1[0]
            rir = rirs.next().split()
            rir_file_location = rir[-1]
            command_list.append("{2} {0} wav-reverberate - {1} - |\n".format(wav_pipe, rir_file_location, output_wav_id))

    file_handle = open(output_dir + "/wav.scp", 'w')
    file_handle.write("".join(command_list))
    file_handle.close()


def Corrupt_wav_noise(input_dir, output_dir, rir_list, noise_list, snr_array, num_replica, prefix):
    rirs = list_cyclic_iterator(return_nonempty_lines(open(rir_list).readlines()), random_seed = 1)
    noises = list_cyclic_iterator(return_nonempty_lines(open(noise_list).readlines()), random_seed = 1)
    snrs = list_cyclic_iterator(snr_array)
    wav_files = open(input_dir + "/wav.scp", 'r').readlines()
    command_list = []
    for i in range(num_replica):
        for j in range(len(wav_files)):
            split1 = wav_files[j].split()
            wav_pipe = " ".join(split1[1:])
            output_wav_id = prefix + str(i) + "_" + split1[0]
            rir = rirs.next().split()
            rir_file_location = rir[-1]
            noise = noises.next().split()
            noise_file_location = noise[-1]
            snr = snrs.next()
            command_list.append("{4} {0} wav-reverberate --noise-file={2} --snr-db={3} - {1} - |\n".format(wav_pipe, rir_file_location, noise_file_location, snr, output_wav_id))

    file_handle = open(output_dir + "/wav.scp", 'w')
    file_handle.write("".join(command_list))
    file_handle.close()



def Replicate_file1(input_file, output_file, num_replica, prefix):
    list = map(lambda x: x.strip(), open(input_file))
    f = open(output_file, "w")
    for i in range(num_replica):
        for line in list:
            split1 = line.split()
            split1[0] = prefix + str(i) + "_" + split1[0]
            print(" ".join(split1), file=f)
    f.close()


def Replicate_file2(input_file, output_file, num_replica, prefix):
    list = map(lambda x: x.strip(), open(input_file))
    f = open(output_file, "w")
    for i in range(num_replica):
        for line in list:
            split1 = line.split()
            split1[0] = prefix + str(i) + "_" + split1[0]
            split1[1] = prefix + str(i) + "_" + split1[1]
            print(" ".join(split1), file=f)
    f.close()


def Main():
    args = GetArgs()

    snr_array = [20 , 15, 10 , 5, 0]
    if args.noise_list is not None:
        Corrupt_wav_noise(args.input_dir, args.output_dir, args.rir_list, args.noise_list, snr_array, args.num_replica, "rvb")
    else:
        Corrupt_wav(args.input_dir, args.output_dir, args.rir_list, args.num_replica, "rvb")
    Replicate_file2(args.input_dir + "/utt2spk", args.output_dir + "/utt2spk", args.num_replica, "rvb")
    train_lib.RunKaldiCommand("utils/utt2spk_to_spk2utt.pl <{output_dir}/utt2spk >{output_dir}/spk2utt"
                    .format(output_dir = args.output_dir))

    if os.path.isfile(args.input_dir + "/text"):
        Replicate_file1(args.input_dir + "/text", args.output_dir + "/text", args.num_replica, "rvb")
    if os.path.isfile(args.input_dir + "/segments"):
        Replicate_file2(args.input_dir + "/segments", args.output_dir + "/segments", args.num_replica, "rvb")
    if os.path.isfile(args.input_dir + "/reco2file_and_channel"):
        Replicate_file2(args.input_dir + "/reco2file_and_channel", args.output_dir + "/reco2file_and_channel", args.num_replica, "rvb")
 
    train_lib.RunKaldiCommand("utils/validate_data_dir.sh --no-feats {output_dir}"
                    .format(output_dir = args.output_dir))

if __name__ == "__main__":
    Main()


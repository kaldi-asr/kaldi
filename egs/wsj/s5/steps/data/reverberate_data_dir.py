#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate reverberated data

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, glob, math, os, random, sys, warnings, copy, imp, ast

train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

class list_cyclic_iterator:
  def __init__(self, list):
    self.list_index = 0
    self.list = list
    random.shuffle(self.list)

  def next(self):
    item = self.list[self.list_index]
    self.list_index = (self.list_index + 1) % len(self.list)
    return item


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Reverberate the data directory with an option "
                                                 "to add isotropic and point source noiseis. "
                                                 "Usage: reverberate_data_dir.py [options...] <in-data-dir> <out-data-dir> "
                                                 "E.g. reverberate_data_dir.py --rir-list-file rir_list "
                                                 "--foreground-snrs 20:10:15:5:0 --background-snrs 20:10:15:5:0 "
                                                 "--noise-list-file noise_list --speech-rvb-probability 1 --num-replications 2 "
                                                 "--random-seed 1 data/train data/train_rvb",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--rir-list-file", type=str, required = True, 
                        help="RIR information file, the format of the file is "
                        "--rir-id <string,compulsary> --room-id <string,compulsary> "
                        "--receiver-position-id <string,optional> --source-position-id <string,optional> "
                        "--rt-60 < <float,optional> --drr <float, optional> < location(support Kaldi IO strings) >")
    parser.add_argument("--noise-list-file", type=str, default = None,
                        help="Noise information file, the format of the file is"
                        "--noise-id <string,compulsary> --noise-type <choices = (isotropic, point source),compulsary> "
                        "--bg-fg-type <choices=(background|foreground), default=background> "
                        "--rir-file <str, compulsary if isotropic, should not be specified if point-source> "
                        "< location=(support Kaldi IO strings) >")
    parser.add_argument("--num-replications", type=int, dest = "num_replicas", default = 1,
                        help="Number of replicate to generated for the data")
    parser.add_argument('--foreground-snrs', type=str, dest = "foreground_snr_string", default = '20:10:0', help='snrs for foreground noises')
    parser.add_argument('--background-snrs', type=str, dest = "background_snr_string", default = '20:10:0', help='snrs for background noises')
    parser.add_argument('--prefix', type=str, default = None, help='prefix for the id of the corrupted utterances')
    parser.add_argument("--speech-rvb-probability", type=float, default = 0.8,
                        help="Probability of reverberating the speech signal, e.g. 0 <= p <= 1")
    parser.add_argument("--noise-adding-probability", type=float, default = 0.4,
                        help="Probability of adding point-source noises, e.g. 0 <= p <= 1")
    parser.add_argument("--max-noises-added", type=int, default = 2,
                        help="Maximum number of point-source noises could be added")
    parser.add_argument('--random-seed', type=int, default=0, help='seed to be used in the randomization of impulese and noises')
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
    if not os.path.isfile(args.rir_list_file):
        raise Exception(args.rir_list_file + "not found")
    
    if args.noise_list_file is not None:
        if not os.path.isfile(args.noise_list_file):
            raise Exception(args.noise_list_file + "not found")

    if args.num_replicas > 1 and args.prefix is None:
        args.prefix = "rvb"
        warnings.warn("--prefix is set to 'rvb' as --num-replications is larger than 1.")

    return args


# This function pick the item according to the associated probability
# The input could be either a dictinoary of a list
def PickItemWithProbability(x):
   if isinstance(x, dict):
     plist = list(set(x.values()))
   else:
     plist = x
   total_p = sum(item.probability for item in plist)
   p = random.uniform(0, total_p)
   accumulate_p = 0
   for item in plist:
      if accumulate_p + item.probability >= p:
         return item
      accumulate_p += item.probability
   assert False, "Shouldn't get here as the accumulated probability should always equal to 1"


def ParseFileToDict(file, assert2fields = False, value_processor = None):
    if value_processor is None:
        value_processor = lambda x: x[0]

    dict = {}
    for line in open(file, 'r'):
        parts = line.split()
        if assert2fields:
            assert(len(parts) == 2)

        dict[parts[0]] = value_processor(parts[1:])
    return dict


# This is the major function to generate pipeline command for the corruption
# The generic command of wav-reverberate will be like:
# wav-reverberate --duration=t --impulse-response=rir.wav 
# --additive-signals='noise1.wav,noise2.wav' --snrs='snr1,snr2' --start-times='s1,s2' input.wav output.wav
def CorruptWav(wav_scp, durations, output_dir, room_dict, noise_list, foreground_snr_array, background_snr_array, num_replicas, prefix, speech_rvb_probability, noise_adding_probability, max_noises_added):
    foreground_snrs = list_cyclic_iterator(foreground_snr_array)
    background_snrs = list_cyclic_iterator(background_snr_array)
    command_list = []
    for i in range(num_replicas):
        keys = wav_scp.keys()
        keys.sort()
        for wav_id in keys:
            wav_pipe = wav_scp[wav_id]
            # check if it is really a pipe
            if len(wav_pipe.split()) == 1:
                wav_pipe = "cat {0} |".format(wav_pipe)
            speech_dur = durations[wav_id]
            if prefix is not None:
                wav_id = prefix + str(i) + "_" + wav_id

            # pick the room
            room = PickItemWithProbability(room_dict)
            command_opts = ""
            noises_added = []
            snrs_added = []
            start_times_added = []
            if random.random() < speech_rvb_probability:
                # pick the RIR to reverberate the speech
                speech_rir = PickItemWithProbability(room.rir_list)
                command_opts += "--impulse-response={0} ".format(speech_rir.rir_file_location)
                # add the corresponding isotropic noise if there is any
                if len(speech_rir.iso_noise_list) > 0:
                    isotropic_noise = PickItemWithProbability(speech_rir.iso_noise_list)
                    # extend the isotropic noise to the length of the speech waveform
                    noises_added.append("wav-reverberate --duration={1} {0} - |".format(isotropic_noise.noise_file_location, speech_dur))
                    snrs_added.append(background_snrs.next())
                    start_times_added.append(0)

            # Add the point-source noise
            if len(noise_list) > 0 and random.random() < noise_adding_probability:
                for k in range(random.randint(1, max_noises_added)):
                    # pick the RIR to reverberate the point-source noise
                    noise = PickItemWithProbability(noise_list)
                    noise_rir = PickItemWithProbability(room.rir_list)
                    if noise.bg_fg_type == "background": 
                        start_times_added.append(0)
                        noises_added.append("wav-reverberate --duration={2} --impulse-response={1} {0} - |".format(noise.noise_file_location, noise_rir.rir_file_location, speech_dur))
                        snrs_added.append(background_snrs.next())
                    else:
                        start_times_added.append(round(random.random() * speech_dur, 2))
                        noises_added.append("wav-reverberate --impulse-response={1} {0} - |".format(noise.noise_file_location, noise_rir.rir_file_location))
                        snrs_added.append(foreground_snrs.next())

            if len(noises_added) > 0:
                command_opts += "--additive-signals='{0}' ".format(','.join(noises_added))
            if len(snrs_added) > 0:
                command_opts += "--snrs='{0}' ".format(','.join(map(lambda x:str(x),snrs_added)))
            if len(start_times_added) > 0:
                command_opts += "--start-times='{0}' ".format(','.join(map(lambda x:str(x),start_times_added)))
            
            if command_opts == "":
                command = "{0} {1}\n".format(wav_id, wav_pipe) 
            else:
                command = "{0} {1} wav-reverberate {2} - - |\n".format(wav_id, wav_pipe, command_opts)

            command_list.append(command)

    file_handle = open(output_dir + "/wav.scp", 'w')
    file_handle.write("".join(command_list))
    file_handle.close()


# This function replicate the entries in files like segments, utt2spk, text
def AddPrefixToFields(input_file, output_file, num_replicas, prefix, field = [0]):
    list = map(lambda x: x.strip(), open(input_file))
    f = open(output_file, "w")
    for i in range(num_replicas):
        for line in list:
            if len(line) > 0 and line[0] != ';':
                split1 = line.split()
                for j in field:
                    if prefix is not None:
                        split1[j] = prefix + str(i) + "_" + split1[j]
                print(" ".join(split1), file=f)
            else:
                print(line, file=f)
    f.close()


def CreateReverberatedCopy(input_dir, output_dir, room_dict, noise_list, foreground_snr_string, background_snr_string, num_replicas, prefix, speech_rvb_probability, noise_adding_probability, max_noises_added):
    
    if not os.path.isfile(input_dir + "/reco2dur"):
        print("Getting the duration of the recordings...");
        train_lib.RunKaldiCommand("wav-to-duration --read-entire-file=true scp:{0}/wav.scp ark,t:{0}/reco2dur".format(input_dir))
    durations = ParseFileToDict(input_dir + "/reco2dur", value_processor = lambda x: float(x[0]))
    wav_scp = ParseFileToDict(input_dir + "/wav.scp", value_processor = lambda x: " ".join(x))
    foreground_snr_array = map(lambda x: float(x), foreground_snr_string.split(':'))
    background_snr_array = map(lambda x: float(x), background_snr_string.split(':'))

    CorruptWav(wav_scp, durations, output_dir, room_dict, noise_list, foreground_snr_array, background_snr_array, num_replicas, prefix, speech_rvb_probability, noise_adding_probability, max_noises_added)

    AddPrefixToFields(input_dir + "/utt2spk", output_dir + "/utt2spk", num_replicas, prefix, field = [0,1])
    train_lib.RunKaldiCommand("utils/utt2spk_to_spk2utt.pl <{output_dir}/utt2spk >{output_dir}/spk2utt"
                    .format(output_dir = output_dir))

    if os.path.isfile(input_dir + "/text"):
        AddPrefixToFields(input_dir + "/text", output_dir + "/text", num_replicas, prefix, field =[0])
    if os.path.isfile(input_dir + "/segments"):
        AddPrefixToFields(input_dir + "/segments", output_dir + "/segments", num_replicas, prefix, field = [0,1])
    if os.path.isfile(input_dir + "/reco2file_and_channel"):
        AddPrefixToFields(input_dir + "/reco2file_and_channel", output_dir + "/reco2file_and_channel", num_replicas, prefix, field = [0,1])

    train_lib.RunKaldiCommand("utils/validate_data_dir.sh --no-feats {output_dir}"
                    .format(output_dir = output_dir))

def SmoothProbability(list):
    uniform_probability = 1 / float(len(list))
    for item in list:
        if item.probability is None:
            item.probability = uniform_probability
        else:
            # smooth the probability
            item.probability = 0.3 * item.probability + 0.7 * uniform_probability

    sum_p = sum(item.probability for item in list)
    # Normalize the probability
    for item in list:
        item.probability = item.probability / sum_p

    return list

def ParseRirList(rir_list_file):
    rir_parser = argparse.ArgumentParser()
    rir_parser.add_argument('--rir-id', type=str, required=True, help='rir id')
    rir_parser.add_argument('--room-id', type=str, required=True, help='room id')
    rir_parser.add_argument('--receiver-position-id', type=str, default=None, help='receiver position id')
    rir_parser.add_argument('--source-position-id', type=str, default=None, help='source position id')
    rir_parser.add_argument('--rt60', type=float, default=None, help='RT60 is the time required for reflections of a direct sound to decay 60 dB.')
    rir_parser.add_argument('--drr', type=float, default=None, help='Direct-to-reverberant-ratio of the impulse.')
    rir_parser.add_argument('--probability', type=float, default=None, help='probability of the impulse.')
    rir_parser.add_argument('rir_file_location', type=str, help='rir file location')

    rir_list = []
    rir_lines = map(lambda x: x.strip(), open(rir_list_file))
    for line in rir_lines:
        rir = rir_parser.parse_args(line.split())
        setattr(rir, "iso_noise_list", [])
        rir_list.append(rir)

    return SmoothProbability(rir_list)


# This function crate the room dictinoary from the rir list
# The key of the returned dictionary is the room id
def MakeRoomDict(rir_list):
    room_dict = {}
    for rir in rir_list:
        if rir.room_id not in room_dict:
            # add new room
            room_dict[rir.room_id] = lambda: None
            setattr(room_dict[rir.room_id], "rir_list", [])
            setattr(room_dict[rir.room_id], "probability", 0)
        room_dict[rir.room_id].rir_list.append(rir)

    for key in room_dict.keys():
        room_dict[key].probability = sum(rir.probability for rir in room_dict[key].rir_list)

    return room_dict


def ParseNoiseList(rir_list, noise_list_file):
    noise_parser = argparse.ArgumentParser()
    noise_parser.add_argument('--noise-id', type=str, required=True, help='noise id')
    noise_parser.add_argument('--noise-type', type=str, required=True, help='the type of noise; i.e. isotropic or point-source', choices = ["isotropic", "point-source"])
    noise_parser.add_argument('--bg-fg-type', type=str, default="background", help='background or foreground noise', choices = ["background", "foreground"])
    noise_parser.add_argument('--rir-file', type=str, default=None, help='compulsary if isotropic, should not be specified if point-source')
    noise_parser.add_argument('--probability', type=float, default=None, help='probability of the noise.')
    noise_parser.add_argument('noise_file_location', type=str, help='noise file location')

    point_noise_list = []
    iso_noise_list = []
    noise_lines = map(lambda x: x.strip(), open(noise_list_file))
    for line in noise_lines:
        noise = noise_parser.parse_args(line.split())
        if noise.noise_type == "isotropic":
            if noise.rir_file is None:
                raise Exception("--rir-file must be specified if --noise-type is point-source")
            else:
                iso_noise_list.append(noise)
        else:
            point_noise_list.append(noise)

    iso_noise_list = SmoothProbability(iso_noise_list)

    for iso_noise in iso_noise_list:
        id = -1
        for j in range(len(rir_list)):
            if iso_noise.rir_file == rir_list[j].rir_file_location:
                id = j
                rir_list[id].iso_noise_list.append(noise)
                break;
        if id == -1:
            warnings.warn("Rir file specified for noise id {0} is not found in rir_list".format(iso_noise.noise_id))

    return (SmoothProbability(point_noise_list), rir_list)


def Main():
    args = GetArgs()
    random.seed(args.random_seed)
    rir_list = ParseRirList(args.rir_list_file)
    noise_list = []
    if args.noise_list_file is not None:
        noise_list, rir_list = ParseNoiseList(rir_list, args.noise_list_file)
        print("Number of point-source noises is {0}".format(len(noise_list)))
    room_dict = MakeRoomDict(rir_list)

    CreateReverberatedCopy(input_dir = args.input_dir,
                   output_dir = args.output_dir,
                   room_dict = room_dict,
                   noise_list = noise_list,
                   foreground_snr_string = args.foreground_snr_string,
                   background_snr_string = args.background_snr_string,
                   num_replicas = args.num_replicas,
                   prefix = args.prefix,
                   speech_rvb_probability = args.speech_rvb_probability,
                   noise_adding_probability = args.noise_adding_probability,
                   max_noises_added = args.max_noises_added)

if __name__ == "__main__":
    Main()


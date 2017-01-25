#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate reverberated data

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, glob, math, os, random, sys, warnings, copy, imp, ast

import data_dir_manipulation_lib as data_lib
sys.path.insert(0, 'steps')
import libs.common as common_lib

def GetArgs():
    # we add required arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Reverberate the data directory with an option "
                                                 "to add isotropic and point source noises. "
                                                 "Usage: reverberate_data_dir.py [options...] <in-data-dir> <out-data-dir> "
                                                 "E.g. reverberate_data_dir.py --rir-set-parameters rir_list "
                                                 "--foreground-snrs 20:10:15:5:0 --background-snrs 20:10:15:5:0 "
                                                 "--noise-list-file noise_list --speech-rvb-probability 1 --num-replications 2 "
                                                 "--random-seed 1 data/train data/train_rvb",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--rir-set-parameters", type=str, action='append', required = True, dest = "rir_set_para_array",
                        help="Specifies the parameters of an RIR set. "
                        "Supports the specification of  mixture_weight and rir_list_file_name. The mixture weight is optional. "
                        "The default mixture weight is the probability mass remaining after adding the mixture weights "
                        "of all the RIR lists, uniformly divided among the RIR lists without mixture weights. "
                        "E.g. --rir-set-parameters '0.3, rir_list' or 'rir_list' "
                        "the format of the RIR list file is "
                        "--rir-id <string,required> --room-id <string,required> "
                        "--receiver-position-id <string,optional> --source-position-id <string,optional> "
                        "--rt-60 <float,optional> --drr <float, optional> location <rspecifier> "
                        "E.g. --rir-id 00001 --room-id 001 --receiver-position-id 001 --source-position-id 00001 "
                        "--rt60 0.58 --drr -4.885 data/impulses/Room001-00001.wav")
    parser.add_argument("--noise-set-parameters", type=str, action='append', default = None, dest = "noise_set_para_array",
                        help="Specifies the parameters of an noise set. "
                        "Supports the specification of mixture_weight and noise_list_file_name. The mixture weight is optional. "
                        "The default mixture weight is the probability mass remaining after adding the mixture weights "
                        "of all the noise lists, uniformly divided among the noise lists without mixture weights. "
                        "E.g. --noise-set-parameters '0.3, noise_list' or 'noise_list' "
                        "the format of the noise list file is "
                        "--noise-id <string,required> --noise-type <choices = {isotropic, point source},required> "
                        "--bg-fg-type <choices = {background, foreground}, default=background> "
                        "--room-linkage <str, specifies the room associated with the noise file. Required if isotropic> "
                        "location <rspecifier> "
                        "E.g. --noise-id 001 --noise-type isotropic --rir-id 00019 iso_noise.wav")
    parser.add_argument("--num-replications", type=int, dest = "num_replicas", default = 1,
                        help="Number of replicate to generated for the data")
    parser.add_argument('--foreground-snrs', type=str, dest = "foreground_snr_string", default = '20:10:0', help='When foreground noises are being added the script will iterate through these SNRs.')
    parser.add_argument('--background-snrs', type=str, dest = "background_snr_string", default = '20:10:0', help='When background noises are being added the script will iterate through these SNRs.')
    parser.add_argument('--prefix', type=str, default = None, help='This prefix will modified for each reverberated copy, by adding additional affixes.')
    parser.add_argument("--speech-rvb-probability", type=float, default = 1.0,
                        help="Probability of reverberating a speech signal, e.g. 0 <= p <= 1")
    parser.add_argument("--pointsource-noise-addition-probability", type=float, default = 1.0,
                        help="Probability of adding point-source noises, e.g. 0 <= p <= 1")
    parser.add_argument("--isotropic-noise-addition-probability", type=float, default = 1.0,
                        help="Probability of adding isotropic noises, e.g. 0 <= p <= 1")
    parser.add_argument("--rir-smoothing-weight", type=float, default = 0.3,
                        help="Smoothing weight for the RIR probabilties, e.g. 0 <= p <= 1. If p = 0, no smoothing will be done. "
                        "The RIR distribution will be mixed with a uniform distribution according to the smoothing weight")
    parser.add_argument("--noise-smoothing-weight", type=float, default = 0.3,
                        help="Smoothing weight for the noise probabilties, e.g. 0 <= p <= 1. If p = 0, no smoothing will be done. "
                        "The noise distribution will be mixed with a uniform distribution according to the smoothing weight")
    parser.add_argument("--max-noises-per-minute", type=int, default = 2,
                        help="This controls the maximum number of point-source noises that could be added to a recording according to its duration")
    parser.add_argument('--random-seed', type=int, default=0, help='seed to be used in the randomization of impulses and noises')
    parser.add_argument("--shift-output", type=str, help="If true, the reverberated waveform will be shifted by the amount of the peak position of the RIR",
                         choices=['true', 'false'], default = "true")
    parser.add_argument('--source-sampling-rate', type=int, default=None,
                        help="Sampling rate of the source data. If a positive integer is specified with this option, "
                        "the RIRs/noises will be resampled to the rate of the source data.")
    parser.add_argument("--include-original-data", type=str, help="If true, the output data includes one copy of the original data",
                         choices=['true', 'false'], default = "false")
    parser.add_argument("--output-additive-noise-dir", type=str,
                        action = common_lib.NullstrToNoneAction, default = None,
                        help="Output directory corresponding to the additive noise part of the data corruption")
    parser.add_argument("--output-reverb-dir", type=str,
                        action = common_lib.NullstrToNoneAction, default = None,
                        help="Output directory corresponding to the reverberated signal part of the data corruption")

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

    if args.prefix is None:
        if args.num_replicas > 1 or args.include_original_data == "true":
            args.prefix = "rvb"
            warnings.warn("--prefix is set to 'rvb' as more than one copy of data is generated")

    if args.output_reverb_dir is not None:
        if not os.path.exists(args.output_reverb_dir):
            os.makedirs(args.output_reverb_dir)

    if args.output_additive_noise_dir is not None:
        if not os.path.exists(args.output_additive_noise_dir):
            os.makedirs(args.output_additive_noise_dir)

    ## Check arguments.

    if args.num_replicas > 1 and args.prefix is None:
        args.prefix = "rvb"
        warnings.warn("--prefix is set to 'rvb' as --num-replications is larger than 1.")

    if not args.num_replicas > 0:
        raise Exception("--num-replications cannot be non-positive")

    if args.speech_rvb_probability < 0 or args.speech_rvb_probability > 1:
        raise Exception("--speech-rvb-probability must be between 0 and 1")

    if args.pointsource_noise_addition_probability < 0 or args.pointsource_noise_addition_probability > 1:
        raise Exception("--pointsource-noise-addition-probability must be between 0 and 1")

    if args.isotropic_noise_addition_probability < 0 or args.isotropic_noise_addition_probability > 1:
        raise Exception("--isotropic-noise-addition-probability must be between 0 and 1")

    if args.rir_smoothing_weight < 0 or args.rir_smoothing_weight > 1:
        raise Exception("--rir-smoothing-weight must be between 0 and 1")

    if args.noise_smoothing_weight < 0 or args.noise_smoothing_weight > 1:
        raise Exception("--noise-smoothing-weight must be between 0 and 1")

    if args.max_noises_per_minute < 0:
        raise Exception("--max-noises-per-minute cannot be negative")

    if args.source_sampling_rate is not None and args.source_sampling_rate <= 0:
        raise Exception("--source-sampling-rate cannot be non-positive")

    return args


# This is the main function to generate pipeline command for the corruption
# The generic command of wav-reverberate will be like:
# wav-reverberate --duration=t --impulse-response=rir.wav
# --additive-signals='noise1.wav,noise2.wav' --snrs='snr1,snr2' --start-times='s1,s2' input.wav output.wav
def GenerateReverberatedWavScp(wav_scp,  # a dictionary whose values are the Kaldi-IO strings of the speech recordings
                               durations, # a dictionary whose values are the duration (in sec) of the speech recordings
                               output_dir, # output directory to write the corrupted wav.scp
                               room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
                               pointsource_noise_list, # the point source noise list
                               iso_noise_dict, # the isotropic noise dictionary
                               foreground_snr_array, # the SNR for adding the foreground noises
                               background_snr_array, # the SNR for adding the background noises
                               num_replicas, # Number of replicate to generated for the data
                               include_original, # include a copy of the original data
                               prefix, # prefix for the id of the corrupted utterances
                               speech_rvb_probability, # Probability of reverberating a speech signal
                               shift_output, # option whether to shift the output waveform
                               isotropic_noise_addition_probability, # Probability of adding isotropic noises
                               pointsource_noise_addition_probability, # Probability of adding point-source noises
                               max_noises_per_minute, # maximum number of point-source noises that can be added to a recording according to its duration
                               output_reverb_dir = None,
                               output_additive_noise_dir = None
                               ):
    foreground_snrs = data_lib.list_cyclic_iterator(foreground_snr_array)
    background_snrs = data_lib.list_cyclic_iterator(background_snr_array)
    corrupted_wav_scp = {}
    reverb_wav_scp = {}
    additive_noise_wav_scp = {}
    keys = wav_scp.keys()
    keys.sort()

    additive_signals_info = {}

    if include_original:
        start_index = 0
    else:
        start_index = 1

    for i in range(start_index, num_replicas+1):
        for recording_id in keys:
            wav_original_pipe = wav_scp[recording_id]
            # check if it is really a pipe
            if len(wav_original_pipe.split()) == 1:
                wav_original_pipe = "cat {0} |".format(wav_original_pipe)
            speech_dur = durations[recording_id]
            max_noises_recording = math.ceil(max_noises_per_minute * speech_dur / 60)

            [impulse_response_opts, noise_addition_descriptor] = data_lib.GenerateReverberationOpts(room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
                                                                                     pointsource_noise_list, # the point source noise list
                                                                                     iso_noise_dict, # the isotropic noise dictionary
                                                                                     foreground_snrs, # the SNR for adding the foreground noises
                                                                                     background_snrs, # the SNR for adding the background noises
                                                                                     speech_rvb_probability, # Probability of reverberating a speech signal
                                                                                     isotropic_noise_addition_probability, # Probability of adding isotropic noises
                                                                                     pointsource_noise_addition_probability, # Probability of adding point-source noises
                                                                                     speech_dur,  # duration of the recording
                                                                                     max_noises_recording  # Maximum number of point-source noises that can be added
                                                                                     )
            additive_noise_opts = ""

            if len(noise_addition_descriptor['noise_io']) > 0:
                additive_noise_opts += "--additive-signals='{0}' ".format(','.join(noise_addition_descriptor['noise_io']))
                additive_noise_opts += "--start-times='{0}' ".format(','.join(map(lambda x:str(x), noise_addition_descriptor['start_times'])))
                additive_noise_opts += "--snrs='{0}' ".format(','.join(map(lambda x:str(x), noise_addition_descriptor['snrs'])))

            reverberate_opts = impulse_response_opts + additive_noise_opts

            new_recording_id = data_lib.GetNewId(recording_id, prefix, i)

            # prefix using index 0 is reserved for original data e.g. rvb0_swb0035 corresponds to the swb0035 recording in original data
            if reverberate_opts == "" or i == 0:
                wav_corrupted_pipe = "{0}".format(wav_original_pipe)
            else:
                wav_corrupted_pipe = "{0} wav-reverberate --shift-output={1} {2} - - |".format(wav_original_pipe, shift_output, reverberate_opts)

            corrupted_wav_scp[new_recording_id] = wav_corrupted_pipe

            if output_reverb_dir is not None:
                if impulse_response_opts == "":
                    wav_reverb_pipe = "{0}".format(wav_original_pipe)
                else:
                    wav_reverb_pipe = "{0} wav-reverberate --shift-output={1} --reverb-out-wxfilename=- {2} - /dev/null |".format(wav_original_pipe, shift_output, reverberate_opts)
                reverb_wav_scp[new_recording_id] = wav_reverb_pipe

            if output_additive_noise_dir is not None:
                if additive_noise_opts != "":
                    wav_additive_noise_pipe = "{0} wav-reverberate --shift-output={1} --additive-noise-out-wxfilename=- {2} - /dev/null |".format(wav_original_pipe, shift_output, reverberate_opts)
                    additive_noise_wav_scp[new_recording_id] = wav_additive_noise_pipe

            if additive_noise_opts != "":
                additive_signals_info[new_recording_id] = [
                        ':'.join(x)
                        for x in zip(noise_addition_descriptor['noise_ids'],
                                     [ str(x) for x in noise_addition_descriptor['start_times'] ],
                                     [ str(x) for x in noise_addition_descriptor['durations'] ])
                        ]

    # Write for each new recording, the id, start time and durations
    # of the signals. Duration is -1 for the foreground noise and needs to
    # be extracted separately if required by determining the durations
    # using the wav file
    data_lib.WriteDictToFile(additive_signals_info, output_dir + "/additive_signals_info.txt")

    data_lib.WriteDictToFile(corrupted_wav_scp, output_dir + "/wav.scp")

    if output_reverb_dir is not None:
        data_lib.WriteDictToFile(reverb_wav_scp, output_reverb_dir + "/wav.scp")

    if output_additive_noise_dir is not None:
        data_lib.WriteDictToFile(additive_noise_wav_scp, output_additive_noise_dir + "/wav.scp")


# This function creates multiple copies of the necessary files, e.g. utt2spk, wav.scp ...
def CreateReverberatedCopy(input_dir,
                           output_dir,
                           room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
                           pointsource_noise_list, # the point source noise list
                           iso_noise_dict, # the isotropic noise dictionary
                           foreground_snr_string, # the SNR for adding the foreground noises
                           background_snr_string, # the SNR for adding the background noises
                           num_replicas, # Number of replicate to generated for the data
                           include_original, # include a copy of the original data
                           prefix, # prefix for the id of the corrupted utterances
                           speech_rvb_probability, # Probability of reverberating a speech signal
                           shift_output, # option whether to shift the output waveform
                           isotropic_noise_addition_probability, # Probability of adding isotropic noises
                           pointsource_noise_addition_probability, # Probability of adding point-source noises
                           max_noises_per_minute,  # maximum number of point-source noises that can be added to a recording according to its duration
                           output_reverb_dir = None,
                           output_additive_noise_dir = None
                           ):

    wav_scp = data_lib.ParseFileToDict(input_dir + "/wav.scp", value_processor = lambda x: " ".join(x))
    if not os.path.isfile(input_dir + "/reco2dur"):
        print("Getting the duration of the recordings...");
        read_entire_file="false"
        for value in wav_scp.values():
            # we will add more checks for sox commands which modify the header as we come across these cases in our data
            if "sox" in value and "speed" in value:
                read_entire_file="true"
                break
        data_lib.RunKaldiCommand("wav-to-duration --read-entire-file={1} scp:{0}/wav.scp ark,t:{0}/reco2dur".format(input_dir, read_entire_file))
    durations = data_lib.ParseFileToDict(input_dir + "/reco2dur", value_processor = lambda x: float(x[0]))
    foreground_snr_array = map(lambda x: float(x), foreground_snr_string.split(':'))
    background_snr_array = map(lambda x: float(x), background_snr_string.split(':'))

    GenerateReverberatedWavScp(wav_scp, durations, output_dir, room_dict, pointsource_noise_list, iso_noise_dict,
                               foreground_snr_array, background_snr_array, num_replicas, include_original, prefix,
                               speech_rvb_probability, shift_output, isotropic_noise_addition_probability,
                               pointsource_noise_addition_probability, max_noises_per_minute,
                               output_reverb_dir = output_reverb_dir,
                               output_additive_noise_dir = output_additive_noise_dir)

    data_lib.CopyDataDirFiles(input_dir, output_dir, num_replicas, include_original, prefix)

    if output_reverb_dir is not None:
        data_lib.CopyDataDirFiles(input_dir, output_reverb_dir, num_replicas, include_original, prefix)

    if output_additive_noise_dir is not None:
        data_lib.CopyDataDirFiles(input_dir, output_additive_noise_dir, num_replicas, include_original, prefix)


def Main():
    args = GetArgs()
    random.seed(args.random_seed)
    rir_list = data_lib.ParseRirList(args.rir_set_para_array, args.rir_smoothing_weight, args.source_sampling_rate)
    print("Number of RIRs is {0}".format(len(rir_list)))
    pointsource_noise_list = []
    iso_noise_dict = {}
    if args.noise_set_para_array is not None:
        pointsource_noise_list, iso_noise_dict = data_lib.ParseNoiseList(args.noise_set_para_array, args.noise_smoothing_weight, args.source_sampling_rate)
        print("Number of point-source noises is {0}".format(len(pointsource_noise_list)))
        print("Number of isotropic noises is {0}".format(sum(len(iso_noise_dict[key]) for key in iso_noise_dict.keys())))
    room_dict = data_lib.MakeRoomDict(rir_list)

    if args.include_original_data == "true":
        include_original = True
    else:
        include_original = False

    CreateReverberatedCopy(input_dir = args.input_dir,
                           output_dir = args.output_dir,
                           room_dict = room_dict,
                           pointsource_noise_list = pointsource_noise_list,
                           iso_noise_dict = iso_noise_dict,
                           foreground_snr_string = args.foreground_snr_string,
                           background_snr_string = args.background_snr_string,
                           num_replicas = args.num_replicas,
                           include_original = include_original,
                           prefix = args.prefix,
                           speech_rvb_probability = args.speech_rvb_probability,
                           shift_output = args.shift_output,
                           isotropic_noise_addition_probability = args.isotropic_noise_addition_probability,
                           pointsource_noise_addition_probability = args.pointsource_noise_addition_probability,
                           max_noises_per_minute = args.max_noises_per_minute,
                           output_reverb_dir = args.output_reverb_dir,
                           output_additive_noise_dir = args.output_additive_noise_dir)

if __name__ == "__main__":
    Main()

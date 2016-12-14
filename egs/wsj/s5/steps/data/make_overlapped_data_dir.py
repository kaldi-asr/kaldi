#!/usr/bin/env python
# Copyright 2016  Tom Ko
# Apache 2.0
# script to generate reverberated data

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse, shlex, glob, math, os, random, sys, warnings, copy, imp, ast

data_lib = imp.load_source('dml', 'steps/data/data_dir_manipulation_lib.py')

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
    parser.add_argument("--noise-set-parameters", type=str, action='append',
                        default = None, dest = "noise_set_para_array",
                        help="Specifies the parameters of an noise set. "
                        "Supports the specification of  mixture_weight and noise_list_file_name. The mixture weight is optional. "
                        "The default mixture weight is the probability mass remaining after adding the mixture weights "
                        "of all the noise lists, uniformly divided among the noise lists without mixture weights. "
                        "E.g. --noise-set-parameters '0.3, noise_list' or 'noise_list' "
                        "the format of the noise list file is "
                        "--noise-id <string,required> --noise-type <choices = {isotropic, point source},required> "
                        "--bg-fg-type <choices = {background, foreground}, default=background> "
                        "--room-linkage <str, specifies the room associated with the noise file. Required if isotropic> "
                        "location <rspecifier> "
                        "E.g. --noise-id 001 --noise-type isotropic --rir-id 00019 iso_noise.wav")
    parser.add_argument("--speech-segments-set-parameters", type=str, action='append',
                        default = None, dest = "speech_segments_set_para_array",
                        help="Specifies the speech segments for overlapped speech generation.\n"
                        "Format: [<probability>], wav_scp, segments_list\n");
    parser.add_argument("--num-replications", type=int, dest = "num_replicas", default = 1,
                        help="Number of replicate to generated for the data")
    parser.add_argument('--foreground-snrs', type=str, dest = "foreground_snr_string",
                        default = '20:10:0',
                        help='When foreground noises are being added the script will iterate through these SNRs.')
    parser.add_argument('--background-snrs', type=str, dest = "background_snr_string",
                        default = '20:10:0',
                        help='When background noises are being added the script will iterate through these SNRs.')
    parser.add_argument('--overlap-snrs', type=str, dest = "overlap_snr_string",
                        default = "20:10:0",
                        help='When overlapping speech segments are being added the script will iterate through these SNRs.')
    parser.add_argument('--prefix', type=str, default = None,
                        help='This prefix will modified for each reverberated copy, by adding additional affixes.')
    parser.add_argument("--speech-rvb-probability", type=float, default = 1.0,
                        help="Probability of reverberating a speech signal, e.g. 0 <= p <= 1")
    parser.add_argument("--pointsource-noise-addition-probability", type=float, default = 1.0,
                        help="Probability of adding point-source noises, e.g. 0 <= p <= 1")
    parser.add_argument("--isotropic-noise-addition-probability", type=float, default = 1.0,
                        help="Probability of adding isotropic noises, e.g. 0 <= p <= 1")
    parser.add_argument("--overlapping-speech-addition-probability", type=float, default = 1.0,
                        help="Probability of adding overlapping speech, e.g. 0 <= p <= 1")
    parser.add_argument("--rir-smoothing-weight", type=float, default = 0.3,
                        help="Smoothing weight for the RIR probabilties, e.g. 0 <= p <= 1. If p = 0, no smoothing will be done. "
                        "The RIR distribution will be mixed with a uniform distribution according to the smoothing weight")
    parser.add_argument("--noise-smoothing-weight", type=float, default = 0.3,
                        help="Smoothing weight for the noise probabilties, e.g. 0 <= p <= 1. If p = 0, no smoothing will be done. "
                        "The noise distribution will be mixed with a uniform distribution according to the smoothing weight")
    parser.add_argument("--overlapping-speech-smoothing-weight", type=float, default = 0.3,
                        help="The overlapping speech distribution will be mixed with a uniform distribution according to the smoothing weight")
    parser.add_argument("--max-noises-per-minute", type=int, default = 2,
                        help="This controls the maximum number of point-source noises that could be added to a recording according to its duration")
    parser.add_argument("--min-overlapping-segments-per-minute", type=int, default = 1,
                        help="This controls the minimum number of overlapping segments of speech that could be added to a recording per minute")
    parser.add_argument("--max-overlapping-segments-per-minute", type=int, default = 5,
                        help="This controls the maximum number of overlapping segments of speech that could be added to a recording per minute")
    parser.add_argument('--random-seed', type=int, default=0,
                        help='seed to be used in the randomization of impulses and noises')
    parser.add_argument("--shift-output", type=str,
                        help="If true, the reverberated waveform will be shifted by the amount of the peak position of the RIR",
                        choices=['true', 'false'], default = "true")
    parser.add_argument("--output-additive-noise-dir", type=str,
                        action = common_train_lib.NullstrToNoneAction, default = None,
                        help="Output directory corresponding to the additive noise part of the data corruption")
    parser.add_argument("--output-reverb-dir", type=str,
                        action = common_train_lib.NullstrToNoneAction, default = None,
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

    if args.output_reverb_dir is not None:
        if args.output_reverb_dir == "":
            args.output_reverb_dir = None

    if args.output_reverb_dir is not None:
        if not os.path.exists(args.output_reverb_dir):
            os.makedirs(args.output_reverb_dir)

    if args.output_additive_noise_dir is not None:
        if args.output_additive_noise_dir == "":
            args.output_additive_noise_dir = None

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

    if args.overlapping_speech_addition_probability < 0 or args.overlapping_speech_addition_probability > 1:
        raise Exception("--overlapping-speech-addition-probability must be between 0 and 1")

    if args.rir_smoothing_weight < 0 or args.rir_smoothing_weight > 1:
        raise Exception("--rir-smoothing-weight must be between 0 and 1")

    if args.noise_smoothing_weight < 0 or args.noise_smoothing_weight > 1:
        raise Exception("--noise-smoothing-weight must be between 0 and 1")

    if args.overlapping_speech_smoothing_weight < 0 or args.overlapping_speech_smoothing_weight > 1:
        raise Exception("--overlapping-speech-smoothing-weight must be between 0 and 1")

    if args.max_noises_per_minute < 0:
        raise Exception("--max-noises-per-minute cannot be negative")

    if args.min_overlapping_segments_per_minute < 0:
        raise Exception("--min-overlapping-segments-per-minute cannot be negative")

    if args.max_overlapping_segments_per_minute < 0:
        raise Exception("--max-overlapping-segments-per-minute cannot be negative")

    return args

def ParseSpeechSegmentsList(speech_segments_set_para_array, smoothing_weight):
    set_list = []
    for set_para in speech_segments_set_para_array:
        set = lambda: None
        setattr(set, "wav_scp", None)
        setattr(set, "segments", None)
        setattr(set, "probability", None)
        parts = set_para.split(',')
        if len(parts) == 3:
            set.probability = float(parts[0])
            set.wav_scp = parts[1].strip()
            set.segments = parts[2].strip()
        else:
            set.wav_scp = parts[0].strip()
            set.segments = parts[1].strip()
        if not os.path.isfile(set.wav_scp):
            raise Exception(set.wav_scp + " not found")
        if not os.path.isfile(set.segments):
            raise Exception(set.segments + " not found")
        set_list.append(set)

    data_lib.SmoothProbabilityDistribution(set_list)

    segments_list = []
    for segments_set in set_list:
        current_segments_list = []

        wav_dict = {}
        for s in open(segments_set.wav_scp):
            parts = s.strip().split()
            wav_dict[parts[0]] = ' '.join(parts[1:])

        for s in open(segments_set.segments):
            parts = s.strip().split()
            current_segment = argparse.Namespace()
            current_segment.utt_id = parts[0]
            current_segment.probability = None

            start_time = float(parts[2])
            end_time = float(parts[3])

            current_segment.duration = (end_time - start_time)

            wav_rxfilename = wav_dict[parts[1]]
            if wav_rxfilename.split()[-1] == '|':
                current_segment.wav_rxfilename = "{0} sox -t wav - -t wav - trim {1} {2} |".format(wav_rxfilename, start_time, end_time - start_time)
            else:
                current_segment.wav_rxfilename = "sox {0} -t wav - trim {1} {2} |".format(wav_rxfilename, start_time, end_time - start_time)

            current_segments_list.append(current_segment)

        segments_list += data_lib.SmoothProbabilityDistribution(current_segments_list, smoothing_weight, segments_set.probability)

    return segments_list

def AddOverlappingSpeech(room,  # the room selected
                         speech_segments_list,    # the speech list
                         overlapping_speech_addition_probability, # Probability of another speech waveform
                         snrs, # the SNR for adding the foreground speech
                         speech_dur,  # duration of the recording
                         min_overlapping_speech_segments,  # Minimum number of speech signals that can be added
                         max_overlapping_speech_segments,  # Maximum number of speech signals that can be added
                         overlapping_speech_descriptor  # descriptor to store the information of the overlapping speech
                        ):
    if (len(speech_segments_list) > 0 and random.random() < overlapping_speech_addition_probability
            and max_overlapping_speech_segments >= 1):
        for k in range(random.randint(min_overlapping_speech_segments, max_overlapping_speech_segments)):
            # pick the overlapping_speech speech signal and the RIR to
            # reverberate the overlapping_speech speech signal
            speech_segment = data_lib.PickItemWithProbability(speech_segments_list)
            rir = data_lib.PickItemWithProbability(room.rir_list)

            speech_rvb_command = """wav-reverberate --impulse-response="{0}" --shift-output=true """.format(rir.rir_rspecifier)
            overlapping_speech_descriptor['start_times'].append(round(random.random() * speech_dur, 2))
            overlapping_speech_descriptor['snrs'].append(snrs.next())
            overlapping_speech_descriptor['utt_ids'].append(speech_segment.utt_id)
            overlapping_speech_descriptor['durations'].append(speech_segment.duration)

            if len(speech_segment.wav_rxfilename.split()) == 1:
                overlapping_speech_descriptor['speech_segments'].append("{1} {0} - |".format(speech_segment.wav_rxfilename, speech_rvb_command))
            else:
                overlapping_speech_descriptor['speech_segments'].append("{0} {1} - - |".format(speech_segment.wav_rxfilename, speech_rvb_command))

# This function randomly decides whether to reverberate, and sample a RIR if it does
# It also decides whether to add the appropriate noises
# This function return the string of options to the binary wav-reverberate
def GenerateReverberationAndOverlappedSpeechOpts(
                              room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
                              pointsource_noise_list, # the point source noise list
                              iso_noise_dict, # the isotropic noise dictionary
                              foreground_snrs, # the SNR for adding the foreground noises
                              background_snrs, # the SNR for adding the background noises
                              speech_segments_list,
                              overlap_snrs,
                              speech_rvb_probability, # Probability of reverberating a speech signal
                              isotropic_noise_addition_probability, # Probability of adding isotropic noises
                              pointsource_noise_addition_probability, # Probability of adding point-source noises
                              overlapping_speech_addition_probability, # Probability of adding overlapping speech segments
                              speech_dur,  # duration of the recording
                              max_noises_recording,  # Maximum number of point-source noises that can be added
                              min_overlapping_segments_recording,  # Minimum number of overlapping segments that can be added
                              max_overlapping_segments_recording   # Maximum number of overlapping segments that can be added
                              ):
    impulse_response_opts = ""

    noise_addition_descriptor = {'noise_io': [],
                                 'start_times': [],
                                 'snrs': [],
                                 'noise_ids': [],
                                 'durations': []
                                 }

    # Randomly select the room
    # Here the room probability is a sum of the probabilities of the RIRs recorded in the room.
    room = data_lib.PickItemWithProbability(room_dict)
    # Randomly select the RIR in the room
    speech_rir = data_lib.PickItemWithProbability(room.rir_list)
    if random.random() < speech_rvb_probability:
        # pick the RIR to reverberate the speech
        impulse_response_opts = """--impulse-response="{0}" """.format(speech_rir.rir_rspecifier)

    rir_iso_noise_list = []
    if speech_rir.room_id in iso_noise_dict:
        rir_iso_noise_list = iso_noise_dict[speech_rir.room_id]
    # Add the corresponding isotropic noise associated with the selected RIR
    if len(rir_iso_noise_list) > 0 and random.random() < isotropic_noise_addition_probability:
        isotropic_noise = data_lib.PickItemWithProbability(rir_iso_noise_list)
        # extend the isotropic noise to the length of the speech waveform
        # check if it is really a pipe
        if len(isotropic_noise.noise_rspecifier.split()) == 1:
            noise_addition_descriptor['noise_io'].append("wav-reverberate --duration={1} {0} - |".format(isotropic_noise.noise_rspecifier, speech_dur))
        else:
            noise_addition_descriptor['noise_io'].append("{0} wav-reverberate --duration={1} - - |".format(isotropic_noise.noise_rspecifier, speech_dur))
        noise_addition_descriptor['start_times'].append(0)
        noise_addition_descriptor['snrs'].append(background_snrs.next())
        noise_addition_descriptor['noise_ids'].append(isotropic_noise.noise_id)
        noise_addition_descriptor['durations'].append(speech_dur)

    data_lib.AddPointSourceNoise(room,  # the room selected
                        pointsource_noise_list, # the point source noise list
                        pointsource_noise_addition_probability, # Probability of adding point-source noises
                        foreground_snrs, # the SNR for adding the foreground noises
                        background_snrs, # the SNR for adding the background noises
                        speech_dur,  # duration of the recording
                        max_noises_recording,  # Maximum number of point-source noises that can be added
                        noise_addition_descriptor  # descriptor to store the information of the noise added
                        )

    assert len(noise_addition_descriptor['noise_io']) == len(noise_addition_descriptor['start_times'])
    assert len(noise_addition_descriptor['noise_io']) == len(noise_addition_descriptor['snrs'])
    assert len(noise_addition_descriptor['noise_io']) == len(noise_addition_descriptor['utt_ids'])
    assert len(noise_addition_descriptor['noise_io']) == len(noise_addition_descriptor['durations'])

    overlapping_speech_descriptor = {'speech_segments': [],
                                     'start_times': [],
                                     'snrs': [],
                                     'utt_ids': [],
                                     'durations': []
                                    }

    print ("Adding overlapping speech...")
    AddOverlappingSpeech(room,
                         speech_segments_list,   # speech segments list
                         overlapping_speech_addition_probability,
                         overlap_snrs,
                         speech_dur,
                         min_overlapping_segments_recording,
                         max_overlapping_segments_recording,
                         overlapping_speech_descriptor
                        )

    return [impulse_response_opts, noise_addition_descriptor,
            overlapping_speech_descriptor]

# This is the main function to generate pipeline command for the corruption
# The generic command of wav-reverberate will be like:
# wav-reverberate --duration=t --impulse-response=rir.wav
# --additive-signals='noise1.wav,noise2.wav' --snrs='snr1,snr2' --start-times='s1,s2' input.wav output.wav
def GenerateReverberatedWavScpWithOverlappedSpeech(
                               wav_scp,  # a dictionary whose values are the Kaldi-IO strings of the speech recordings
                               durations, # a dictionary whose values are the duration (in sec) of the speech recordings
                               output_dir, # output directory to write the corrupted wav.scp
                               room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
                               pointsource_noise_list, # the point source noise list
                               iso_noise_dict, # the isotropic noise dictionary
                               foreground_snr_array, # the SNR for adding the foreground noises
                               background_snr_array, # the SNR for adding the background noises
                               speech_segments_list, # list of speech segments to create overlapped speech
                               overlap_snr_array,   # the SNR for adding overlapping speech
                               num_replicas, # Number of replicate to generated for the data
                               prefix, # prefix for the id of the corrupted utterances
                               speech_rvb_probability, # Probability of reverberating a speech signal
                               shift_output, # option whether to shift the output waveform
                               isotropic_noise_addition_probability, # Probability of adding isotropic noises
                               pointsource_noise_addition_probability, # Probability of adding point-source noises
                               max_noises_per_minute, # maximum number of point-source noises that can be added to a recording according to its duration
                               overlapping_speech_addition_probability,
                               min_overlapping_segments_per_minute,
                               max_overlapping_segments_per_minute,
                               output_reverb_dir = None,
                               output_additive_noise_dir = None,
                               ):
    foreground_snrs = data_lib.list_cyclic_iterator(foreground_snr_array)
    background_snrs = data_lib.list_cyclic_iterator(background_snr_array)
    overlap_snrs = data_lib.list_cyclic_iterator(overlap_snr_array)

    corrupted_wav_scp = {}
    reverb_wav_scp = {}
    additive_noise_wav_scp = {}
    overlapping_segments_info = {}

    keys = wav_scp.keys()
    keys.sort()
    for i in range(1, num_replicas+1):
        for recording_id in keys:
            wav_original_pipe = wav_scp[recording_id]
            # check if it is really a pipe
            if len(wav_original_pipe.split()) == 1:
                wav_original_pipe = "cat {0} |".format(wav_original_pipe)
            speech_dur = durations[recording_id]
            max_noises_recording = math.floor(max_noises_per_minute * speech_dur / 60)
            min_overlapping_segments_recording = max(math.floor(min_overlapping_segments_per_minute * speech_dur / 60), 1)
            max_overlapping_segments_recording = math.floor(max_overlapping_segments_per_minute * speech_dur / 60)

            [impulse_response_opts, noise_addition_descriptor,
             overlapping_speech_descriptor] = GenerateReverberationAndOverlappedSpeechOpts(
                     room_dict = room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
                     pointsource_noise_list = pointsource_noise_list, # the point source noise list
                     iso_noise_dict = iso_noise_dict, # the isotropic noise dictionary
                     foreground_snrs = foreground_snrs, # the SNR for adding the foreground noises
                     background_snrs = background_snrs, # the SNR for adding the background noises
                     speech_segments_list = speech_segments_list,  # Speech segments for creating overlapped speech
                     overlap_snrs = overlap_snrs,  # the SNR for adding overlapping speech
                     speech_rvb_probability = speech_rvb_probability, # Probability of reverberating a speech signal
                     isotropic_noise_addition_probability = isotropic_noise_addition_probability, # Probability of adding isotropic noises
                     pointsource_noise_addition_probability = pointsource_noise_addition_probability, # Probability of adding point-source noises
                     overlapping_speech_addition_probability = overlapping_speech_addition_probability,
                     speech_dur = speech_dur,  # duration of the recording
                     max_noises_recording = max_noises_recording,  # Maximum number of point-source noises that can be added
                     min_overlapping_segments_recording = min_overlapping_segments_recording,
                     max_overlapping_segments_recording = max_overlapping_segments_recording
                     )

            additive_noise_opts = ""

            if (len(noise_addition_descriptor['noise_io']) > 0 or
                    len(overlapping_speech_descriptor['speech_segments']) > 0):
                additive_noise_opts += ("--additive-signals='{0}' "
                                        .format(','
                                        .join(noise_addition_descriptor['noise_io'] +
                                              overlapping_speech_descriptor['speech_segments']))
                                        )
                additive_noise_opts += ("--start-times='{0}' "
                                        .format(','
                                        .join(map(lambda x:str(x), noise_addition_descriptor['start_times'] +
                                                                   overlapping_speech_descriptor['start_times'])))
                                        )
                additive_noise_opts += ("--snrs='{0}' "
                                        .format(','
                                        .join(map(lambda x:str(x), noise_addition_descriptor['snrs'] +
                                                                   overlapping_speech_descriptor['snrs'])))
                                        )

            reverberate_opts = impulse_response_opts + additive_noise_opts

            new_recording_id = data_lib.GetNewId(recording_id, prefix, i)

            if reverberate_opts == "":
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

            if len(overlapping_speech_descriptor['speech_segments']) > 0:
                overlapping_segments_info[new_recording_id] = [
                        ':'.join(x)
                        for x in zip(overlapping_speech_descriptor['utt_ids'],
                                     [ str(x) for x in overlapping_speech_descriptor['start_times'] ],
                                     [ str(x) for x in overlapping_speech_descriptor['durations'] ])
                        ]

    data_lib.WriteDictToFile(corrupted_wav_scp, output_dir + "/wav.scp")

    # Write for each new recording, the id, start time and durations
    # of the overlapping segments
    data_lib.WriteDictToFile(overlapping_segments_info, output_dir + "/overlapped_segments_info.txt")

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
                           speech_segments_list,
                           foreground_snr_string, # the SNR for adding the foreground noises
                           background_snr_string, # the SNR for adding the background noises
                           overlap_snr_string,  # the SNR for overlapping speech
                           num_replicas, # Number of replicate to generated for the data
                           prefix, # prefix for the id of the corrupted utterances
                           speech_rvb_probability, # Probability of reverberating a speech signal
                           shift_output, # option whether to shift the output waveform
                           isotropic_noise_addition_probability, # Probability of adding isotropic noises
                           pointsource_noise_addition_probability, # Probability of adding point-source noises
                           max_noises_per_minute,  # maximum number of point-source noises that can be added to a recording according to its duration
                           overlapping_speech_addition_probability,
                           min_overlapping_segments_per_minute,
                           max_overlapping_segments_per_minute,
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
    overlap_snr_array = map(lambda x: float(x), overlap_snr_string.split(':'))

    GenerateReverberatedWavScpWithOverlappedSpeech(
               wav_scp = wav_scp,
               durations = durations,
               output_dir = output_dir,
               room_dict = room_dict,
               pointsource_noise_list = pointsource_noise_list,
               iso_noise_dict = iso_noise_dict,
               foreground_snr_array = foreground_snr_array,
               background_snr_array = background_snr_array,
               speech_segments_list = speech_segments_list,
               overlap_snr_array = overlap_snr_array,
               num_replicas = num_replicas, prefix = prefix,
               speech_rvb_probability = speech_rvb_probability,
               shift_output = shift_output,
               isotropic_noise_addition_probability = isotropic_noise_addition_probability,
               pointsource_noise_addition_probability =  pointsource_noise_addition_probability,
               max_noises_per_minute = max_noises_per_minute,
               overlapping_speech_addition_probability = overlapping_speech_addition_probability,
               min_overlapping_segments_per_minute = min_overlapping_segments_per_minute,
               max_overlapping_segments_per_minute = max_overlapping_segments_per_minute,
               output_reverb_dir = output_reverb_dir,
               output_additive_noise_dir = output_additive_noise_dir)

    data_lib.CopyDataDirFiles(input_dir, output_dir, num_replicas, prefix)
    data_lib.AddPrefixToFields(input_dir + "/reco2dur", output_dir + "/reco2dur", num_replicas, prefix, field = [0])

    if output_reverb_dir is not None:
        data_lib.CopyDataDirFiles(input_dir, output_reverb_dir, num_replicas, prefix)
        data_lib.AddPrefixToFields(input_dir + "/reco2dur", output_reverb_dir + "/reco2dur", num_replicas, prefix, field = [0])

    if output_additive_noise_dir is not None:
        data_lib.CopyDataDirFiles(input_dir, output_additive_noise_dir, num_replicas, prefix)
        data_lib.AddPrefixToFields(input_dir + "/reco2dur", output_additive_noise_dir + "/reco2dur", num_replicas, prefix, field = [0])


def Main():
    args = GetArgs()
    random.seed(args.random_seed)
    rir_list = data_lib.ParseRirList(args.rir_set_para_array, args.rir_smoothing_weight)
    print("Number of RIRs is {0}".format(len(rir_list)))
    pointsource_noise_list = []
    iso_noise_dict = {}
    if args.noise_set_para_array is not None:
        pointsource_noise_list, iso_noise_dict = data_lib.ParseNoiseList(args.noise_set_para_array, args.noise_smoothing_weight)
        print("Number of point-source noises is {0}".format(len(pointsource_noise_list)))
        print("Number of isotropic noises is {0}".format(sum(len(iso_noise_dict[key]) for key in iso_noise_dict.keys())))
    room_dict = data_lib.MakeRoomDict(rir_list)

    speech_segments_list = ParseSpeechSegmentsList(args.speech_segments_set_para_array, args.overlapping_speech_smoothing_weight)

    CreateReverberatedCopy(input_dir = args.input_dir,
                           output_dir = args.output_dir,
                           room_dict = room_dict,
                           pointsource_noise_list = pointsource_noise_list,
                           iso_noise_dict = iso_noise_dict,
                           speech_segments_list = speech_segments_list,
                           foreground_snr_string = args.foreground_snr_string,
                           background_snr_string = args.background_snr_string,
                           overlap_snr_string = args.overlap_snr_string,
                           num_replicas = args.num_replicas,
                           prefix = args.prefix,
                           speech_rvb_probability = args.speech_rvb_probability,
                           shift_output = args.shift_output,
                           isotropic_noise_addition_probability = args.isotropic_noise_addition_probability,
                           pointsource_noise_addition_probability = args.pointsource_noise_addition_probability,
                           max_noises_per_minute = args.max_noises_per_minute,
                           overlapping_speech_addition_probability = args.overlapping_speech_addition_probability,
                           min_overlapping_segments_per_minute = args.min_overlapping_segments_per_minute,
                           max_overlapping_segments_per_minute = args.max_overlapping_segments_per_minute,
                           output_reverb_dir = args.output_reverb_dir,
                           output_additive_noise_dir = args.output_additive_noise_dir)

if __name__ == "__main__":
    Main()



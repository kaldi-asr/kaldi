#!/usr/bin/env python3
# Copyright 2016  Tom Ko
#           2018  David Snyder
#           2019  Phani Sankar Nidadavolu
# Apache 2.0
# script to generate reverberated data

import argparse, shlex, glob, math, os, random, sys, warnings, copy, imp, ast

data_lib = imp.load_source('dml', 'steps/data/data_dir_manipulation_lib.py')

def get_args():
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
    parser.add_argument("input_dir",
                        help="Input data directory")
    parser.add_argument("output_dir",
                        help="Output data directory")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = check_args(args)

    return args

def check_args(args):
    if args.prefix is None:
        if args.num_replicas > 1 or args.include_original_data == "true":
            args.prefix = "rvb"
            warnings.warn("--prefix is set to 'rvb' as more than one copy of data is generated")

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


class list_cyclic_iterator(object):
    def __init__(self, list):
        self.list_index = 0
        self.list = list
        random.shuffle(self.list)

    def __next__(self):
        item = self.list[self.list_index]
        self.list_index = (self.list_index + 1) % len(self.list)
        return item

    next = __next__  # for Python 2

def pick_item_with_probability(x):
    """ This functions picks an item from the collection according to the associated
        probability distribution. The probability estimate of each item in the collection
        is stored in the "probability" field of the particular item. x : a
        collection (list or dictionary) where the values contain a field called probability
    """
    if isinstance(x, dict):
        keylist = list(x.keys())
        keylist.sort()
        random.shuffle(keylist)
        plist = [x[k] for k in keylist]
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


def parse_file_to_dict(file, assert2fields = False, value_processor = None):
    """ This function parses a file and pack the data into a dictionary
        It is useful for parsing file like wav.scp, utt2spk, text...etc
    """
    if value_processor is None:
        value_processor = lambda x: x[0]
    dict = {}
    for line in open(file, 'r', encoding='utf-8'):
        parts = line.split()
        if assert2fields:
            assert(len(parts) == 2)

        dict[parts[0]] = value_processor(parts[1:])
    return dict

def write_dict_to_file(dict, file_name):
    """ This function creates a file and write the content of a dictionary into it
    """
    file = open(file_name, 'w', encoding='utf-8')
    keys = sorted(dict.keys())
    for key in keys:
        value = dict[key]
        if type(value) in [list, tuple] :
            if type(value) is tuple:
                value = list(value)
            value = sorted(value)
            value = ' '.join(str(value))
        file.write('{0} {1}\n'.format(key, value))
    file.close()


def create_corrupted_utt2uniq(input_dir, output_dir, num_replicas, include_original, prefix):
    """This function creates the utt2uniq file from the utterance id in utt2spk file
    """
    corrupted_utt2uniq = {}
    # Parse the utt2spk to get the utterance id
    utt2spk = parse_file_to_dict(input_dir + "/utt2spk", value_processor = lambda x: " ".join(x))
    keys = sorted(utt2spk.keys())
    if include_original:
        start_index = 0
    else:
        start_index = 1

    for i in range(start_index, num_replicas+1):
        for utt_id in keys:
            new_utt_id = get_new_id(utt_id, prefix, i)
            corrupted_utt2uniq[new_utt_id] = utt_id

    write_dict_to_file(corrupted_utt2uniq, output_dir + "/utt2uniq")


def add_point_source_noise(noise_addition_descriptor,  # descriptor to store the information of the noise added
                        room,  # the room selected
                        pointsource_noise_list, # the point source noise list
                        pointsource_noise_addition_probability, # Probability of adding point-source noises
                        foreground_snrs, # the SNR for adding the foreground noises
                        background_snrs, # the SNR for adding the background noises
                        speech_dur,  # duration of the recording
                        max_noises_recording  # Maximum number of point-source noises that can be added
                        ):
    if len(pointsource_noise_list) > 0 and random.random() < pointsource_noise_addition_probability and max_noises_recording >= 1:
        for k in range(random.randint(1, max_noises_recording)):
            # pick the RIR to reverberate the point-source noise
            noise = pick_item_with_probability(pointsource_noise_list)
            noise_rir = pick_item_with_probability(room.rir_list)
            # If it is a background noise, the noise will be extended and be added to the whole speech
            # if it is a foreground noise, the noise will not extended and be added at a random time of the speech
            if noise.bg_fg_type == "background":
                noise_rvb_command = """wav-reverberate --impulse-response="{0}" --duration={1}""".format(noise_rir.rir_rspecifier, speech_dur)
                noise_addition_descriptor['start_times'].append(0)
                noise_addition_descriptor['snrs'].append(next(background_snrs))
            else:
                noise_rvb_command = """wav-reverberate --impulse-response="{0}" """.format(noise_rir.rir_rspecifier)
                noise_addition_descriptor['start_times'].append(round(random.random() * speech_dur, 2))
                noise_addition_descriptor['snrs'].append(next(foreground_snrs))

            # check if the rspecifier is a pipe or not
            if len(noise.noise_rspecifier.split()) == 1:
                noise_addition_descriptor['noise_io'].append("{1} {0} - |".format(noise.noise_rspecifier, noise_rvb_command))
            else:
                noise_addition_descriptor['noise_io'].append("{0} {1} - - |".format(noise.noise_rspecifier, noise_rvb_command))

    return noise_addition_descriptor


def generate_reverberation_opts(room_dict,  # the room dictionary, please refer to make_room_dict() for the format
                              pointsource_noise_list, # the point source noise list
                              iso_noise_dict, # the isotropic noise dictionary
                              foreground_snrs, # the SNR for adding the foreground noises
                              background_snrs, # the SNR for adding the background noises
                              speech_rvb_probability, # Probability of reverberating a speech signal
                              isotropic_noise_addition_probability, # Probability of adding isotropic noises
                              pointsource_noise_addition_probability, # Probability of adding point-source noises
                              speech_dur,  # duration of the recording
                              max_noises_recording  # Maximum number of point-source noises that can be added
                              ):
    """ This function randomly decides whether to reverberate, and sample a RIR if it does
        It also decides whether to add the appropriate noises
        This function return the string of options to the binary wav-reverberate
    """
    reverberate_opts = ""
    noise_addition_descriptor = {'noise_io': [],
                                 'start_times': [],
                                 'snrs': []}
    # Randomly select the room
    # Here the room probability is a sum of the probabilities of the RIRs recorded in the room.
    room = pick_item_with_probability(room_dict)
    # Randomly select the RIR in the room
    speech_rir = pick_item_with_probability(room.rir_list)
    if random.random() < speech_rvb_probability:
        # pick the RIR to reverberate the speech
        reverberate_opts += """--impulse-response="{0}" """.format(speech_rir.rir_rspecifier)

    rir_iso_noise_list = []
    if speech_rir.room_id in iso_noise_dict:
        rir_iso_noise_list = iso_noise_dict[speech_rir.room_id]
    # Add the corresponding isotropic noise associated with the selected RIR
    if len(rir_iso_noise_list) > 0 and random.random() < isotropic_noise_addition_probability:
        isotropic_noise = pick_item_with_probability(rir_iso_noise_list)
        # extend the isotropic noise to the length of the speech waveform
        # check if the rspecifier is a pipe or not
        if len(isotropic_noise.noise_rspecifier.split()) == 1:
            noise_addition_descriptor['noise_io'].append("wav-reverberate --duration={1} {0} - |".format(isotropic_noise.noise_rspecifier, speech_dur))
        else:
            noise_addition_descriptor['noise_io'].append("{0} wav-reverberate --duration={1} - - |".format(isotropic_noise.noise_rspecifier, speech_dur))
        noise_addition_descriptor['start_times'].append(0)
        noise_addition_descriptor['snrs'].append(next(background_snrs))

    noise_addition_descriptor = add_point_source_noise(noise_addition_descriptor,  # descriptor to store the information of the noise added
                                                    room,  # the room selected
                                                    pointsource_noise_list, # the point source noise list
                                                    pointsource_noise_addition_probability, # Probability of adding point-source noises
                                                    foreground_snrs, # the SNR for adding the foreground noises
                                                    background_snrs, # the SNR for adding the background noises
                                                    speech_dur,  # duration of the recording
                                                    max_noises_recording  # Maximum number of point-source noises that can be added
                                                    )

    assert len(noise_addition_descriptor['noise_io']) == len(noise_addition_descriptor['start_times'])
    assert len(noise_addition_descriptor['noise_io']) == len(noise_addition_descriptor['snrs'])
    if len(noise_addition_descriptor['noise_io']) > 0:
        reverberate_opts += "--additive-signals='{0}' ".format(','.join(noise_addition_descriptor['noise_io']))
        reverberate_opts += "--start-times='{0}' ".format(','.join([str(x) for x in noise_addition_descriptor['start_times']]))
        reverberate_opts += "--snrs='{0}' ".format(','.join([str(x) for x in noise_addition_descriptor['snrs']]))

    return reverberate_opts

def get_new_id(id, prefix=None, copy=0):
    """ This function generates a new id from the input id
        This is needed when we have to create multiple copies of the original data
        E.g. get_new_id("swb0035", prefix="rvb", copy=1) returns a string "rvb1-swb0035"
    """
    if prefix is not None:
        new_id = prefix + str(copy) + "-" + id
    else:
        new_id = id

    return new_id


def generate_reverberated_wav_scp(wav_scp,  # a dictionary whose values are the Kaldi-IO strings of the speech recordings
                               durations, # a dictionary whose values are the duration (in sec) of the speech recordings
                               output_dir, # output directory to write the corrupted wav.scp
                               room_dict,  # the room dictionary, please refer to make_room_dict() for the format
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
                               max_noises_per_minute # maximum number of point-source noises that can be added to a recording according to its duration
                               ):
    """ This is the main function to generate pipeline command for the corruption
        The generic command of wav-reverberate will be like:
        wav-reverberate --duration=t --impulse-response=rir.wav
        --additive-signals='noise1.wav,noise2.wav' --snrs='snr1,snr2' --start-times='s1,s2' input.wav output.wav
    """
    foreground_snrs = list_cyclic_iterator(foreground_snr_array)
    background_snrs = list_cyclic_iterator(background_snr_array)
    corrupted_wav_scp = {}
    keys = sorted(wav_scp.keys())
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
            max_noises_recording = math.floor(max_noises_per_minute * speech_dur / 60)

            reverberate_opts = generate_reverberation_opts(room_dict,  # the room dictionary, please refer to make_room_dict() for the format
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

            # prefix using index 0 is reserved for original data e.g. rvb0_swb0035 corresponds to the swb0035 recording in original data
            if reverberate_opts == "" or i == 0:
                wav_corrupted_pipe = "{0}".format(wav_original_pipe)
            else:
                wav_corrupted_pipe = "{0} wav-reverberate --shift-output={1} {2} - - |".format(wav_original_pipe, shift_output, reverberate_opts)

            new_recording_id = get_new_id(recording_id, prefix, i)
            corrupted_wav_scp[new_recording_id] = wav_corrupted_pipe

    write_dict_to_file(corrupted_wav_scp, output_dir + "/wav.scp")


def add_prefix_to_fields(input_file, output_file, num_replicas, include_original, prefix, field = [0]):
    """ This function replicate the entries in files like segments, utt2spk, text
    """
    list = [x.strip() for x in open(input_file, encoding='utf-8')]
    f = open(output_file, "w", encoding='utf-8')
    if include_original:
        start_index = 0
    else:
        start_index = 1

    for i in range(start_index, num_replicas+1):
        for line in list:
            if len(line) > 0 and line[0] != ';':
                split1 = line.split()
                for j in field:
                    split1[j] = get_new_id(split1[j], prefix, i)
                print(" ".join(split1), file=f)
            else:
                print(line, file=f)
    f.close()


def create_reverberated_copy(input_dir,
                           output_dir,
                           room_dict,  # the room dictionary, please refer to make_room_dict() for the format
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
                           max_noises_per_minute  # maximum number of point-source noises that can be added to a recording according to its duration
                           ):
    """ This function creates multiple copies of the necessary files,
        e.g. utt2spk, wav.scp ...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wav_scp = parse_file_to_dict(input_dir + "/wav.scp", value_processor = lambda x: " ".join(x))
    if not os.path.isfile(input_dir + "/reco2dur"):
        print("Getting the duration of the recordings...");
        data_lib.RunKaldiCommand("utils/data/get_reco2dur.sh {}".format(input_dir))
    durations = parse_file_to_dict(input_dir + "/reco2dur", value_processor = lambda x: float(x[0]))
    foreground_snr_array = [float(x) for x in foreground_snr_string.split(':')]
    background_snr_array = [float(x) for x in background_snr_string.split(':')]

    generate_reverberated_wav_scp(wav_scp, durations, output_dir, room_dict, pointsource_noise_list, iso_noise_dict,
               foreground_snr_array, background_snr_array, num_replicas, include_original, prefix,
               speech_rvb_probability, shift_output, isotropic_noise_addition_probability,
               pointsource_noise_addition_probability, max_noises_per_minute)

    add_prefix_to_fields(input_dir + "/utt2spk", output_dir + "/utt2spk", num_replicas, include_original, prefix, field = [0,1])
    data_lib.RunKaldiCommand("utils/utt2spk_to_spk2utt.pl <{output_dir}/utt2spk >{output_dir}/spk2utt"
                    .format(output_dir = output_dir))

    if os.path.isfile(input_dir + "/utt2uniq"):
        add_prefix_to_fields(input_dir + "/utt2uniq", output_dir + "/utt2uniq", num_replicas, include_original, prefix, field =[0])
    else:
        # Create the utt2uniq file
        create_corrupted_utt2uniq(input_dir, output_dir, num_replicas, include_original, prefix)

    if os.path.isfile(input_dir + "/text"):
        add_prefix_to_fields(input_dir + "/text", output_dir + "/text", num_replicas, include_original, prefix, field =[0])
    if os.path.isfile(input_dir + "/segments"):
        add_prefix_to_fields(input_dir + "/segments", output_dir + "/segments", num_replicas, include_original, prefix, field = [0,1])
    if os.path.isfile(input_dir + "/reco2file_and_channel"):
        add_prefix_to_fields(input_dir + "/reco2file_and_channel", output_dir + "/reco2file_and_channel", num_replicas, include_original, prefix, field = [0,1])
    if os.path.isfile(input_dir + "/vad.scp"):
        add_prefix_to_fields(input_dir + "/vad.scp", output_dir + "/vad.scp", num_replicas, include_original, prefix, field=[0])

    data_lib.RunKaldiCommand("utils/validate_data_dir.sh --no-feats --no-text {output_dir}"
                    .format(output_dir = output_dir))


def smooth_probability_distribution(set_list, smoothing_weight=0.0, target_sum=1.0):
    """ This function smooths the probability distribution in the list
    """
    if len(list(set_list)) > 0:
      num_unspecified = 0
      accumulated_prob = 0
      for item in set_list:
          if item.probability is None:
              num_unspecified += 1
          else:
              accumulated_prob += item.probability

      # Compute the probability for the items without specifying their probability
      uniform_probability = 0
      if num_unspecified > 0 and accumulated_prob < 1:
          uniform_probability = (1 - accumulated_prob) / float(num_unspecified)
      elif num_unspecified > 0 and accumulated_prob >= 1:
          warnings.warn("The sum of probabilities specified by user is larger than or equal to 1. "
                        "The items without probabilities specified will be given zero to their probabilities.")

      for item in set_list:
          if item.probability is None:
              item.probability = uniform_probability
          else:
              # smooth the probability
              item.probability = (1 - smoothing_weight) * item.probability + smoothing_weight * uniform_probability

      # Normalize the probability
      sum_p = sum(item.probability for item in set_list)
      for item in set_list:
          item.probability = item.probability / sum_p * target_sum

    return set_list


def parse_set_parameter_strings(set_para_array):
    """ This function parse the array of rir set parameter strings.
        It will assign probabilities to those rir sets which don't have a probability
        It will also check the existence of the rir list files.
    """
    set_list = []
    for set_para in set_para_array:
        set = lambda: None
        setattr(set, "filename", None)
        setattr(set, "probability", None)
        parts = set_para.split(',')
        if len(parts) == 2:
            set.probability = float(parts[0])
            set.filename = parts[1].strip()
        else:
            set.filename = parts[0].strip()
        if not os.path.isfile(set.filename):
            raise Exception(set.filename + " not found")
        set_list.append(set)

    return smooth_probability_distribution(set_list)


def parse_rir_list(rir_set_para_array, smoothing_weight, sampling_rate = None):
    """ This function creates the RIR list
        Each rir object in the list contains the following attributes:
        rir_id, room_id, receiver_position_id, source_position_id, rt60, drr, probability
        Please refer to the help messages in the parser for the meaning of these attributes
    """
    rir_parser = argparse.ArgumentParser()
    rir_parser.add_argument('--rir-id', type=str, required=True, help='This id is unique for each RIR and the noise may associate with a particular RIR by refering to this id')
    rir_parser.add_argument('--room-id', type=str, required=True, help='This is the room that where the RIR is generated')
    rir_parser.add_argument('--receiver-position-id', type=str, default=None, help='receiver position id')
    rir_parser.add_argument('--source-position-id', type=str, default=None, help='source position id')
    rir_parser.add_argument('--rt60', type=float, default=None, help='RT60 is the time required for reflections of a direct sound to decay 60 dB.')
    rir_parser.add_argument('--drr', type=float, default=None, help='Direct-to-reverberant-ratio of the impulse response.')
    rir_parser.add_argument('--cte', type=float, default=None, help='Early-to-late index of the impulse response.')
    rir_parser.add_argument('--probability', type=float, default=None, help='probability of the impulse response.')
    rir_parser.add_argument('rir_rspecifier', type=str, help="""rir rspecifier, it can be either a filename or a piped command.
                            E.g. data/impulses/Room001-00001.wav or "sox data/impulses/Room001-00001.wav -t wav - |" """)

    set_list = parse_set_parameter_strings(rir_set_para_array)

    rir_list = []
    for rir_set in set_list:
        current_rir_list = [rir_parser.parse_args(shlex.split(x.strip())) for x in open(rir_set.filename)]
        for rir in current_rir_list:
            if sampling_rate is not None:
                # check if the rspecifier is a pipe or not
                if len(rir.rir_rspecifier.split()) == 1:
                    rir.rir_rspecifier = "sox {0} -r {1} -t wav - |".format(rir.rir_rspecifier, sampling_rate)
                else:
                    rir.rir_rspecifier = "{0} sox -t wav - -r {1} -t wav - |".format(rir.rir_rspecifier, sampling_rate)

        rir_list += smooth_probability_distribution(current_rir_list, smoothing_weight, rir_set.probability)

    return rir_list


def almost_equal(value_1, value_2, accuracy = 10**-8):
    """ This function checks if the inputs are approximately equal assuming they are floats.
    """
    return abs(value_1 - value_2) < accuracy


def make_room_dict(rir_list):
    """ This function converts a list of RIRs into a dictionary of RIRs indexed by the room-id.
        Its values are objects with two attributes: a local RIR list
        and the probability of the corresponding room
        Please look at the comments at parse_rir_list() for the attributes that a RIR object contains
    """
    room_dict = {}
    for rir in rir_list:
        if rir.room_id not in room_dict:
            # add new room
            room_dict[rir.room_id] = lambda: None
            setattr(room_dict[rir.room_id], "rir_list", [])
            setattr(room_dict[rir.room_id], "probability", 0)
        room_dict[rir.room_id].rir_list.append(rir)

    # the probability of the room is the sum of probabilities of its RIR
    for key in room_dict.keys():
        room_dict[key].probability = sum(rir.probability for rir in room_dict[key].rir_list)

    assert almost_equal(sum(room_dict[key].probability for key in room_dict.keys()), 1.0)

    return room_dict

def parse_noise_list(noise_set_para_array, smoothing_weight, sampling_rate = None):
    """ This function creates the point-source noise list
         and the isotropic noise dictionary from the noise information file
         The isotropic noise dictionary is indexed by the room
         and its value is the corrresponding isotropic noise list
         Each noise object in the list contains the following attributes:
         noise_id, noise_type, bg_fg_type, room_linkage, probability, noise_rspecifier
         Please refer to the help messages in the parser for the meaning of these attributes
    """
    noise_parser = argparse.ArgumentParser()
    noise_parser.add_argument('--noise-id', type=str, required=True, help='noise id')
    noise_parser.add_argument('--noise-type', type=str, required=True, help='the type of noise; i.e. isotropic or point-source', choices = ["isotropic", "point-source"])
    noise_parser.add_argument('--bg-fg-type', type=str, default="background", help='background or foreground noise, for background noises, '
                              'they will be extended before addition to cover the whole speech; for foreground noise, they will be kept '
                              'to their original duration and added at a random point of the speech.', choices = ["background", "foreground"])
    noise_parser.add_argument('--room-linkage', type=str, default=None, help='required if isotropic, should not be specified if point-source.')
    noise_parser.add_argument('--probability', type=float, default=None, help='probability of the noise.')
    noise_parser.add_argument('noise_rspecifier', type=str, help="""noise rspecifier, it can be either a filename or a piped command.
                              E.g. type5_noise_cirline_ofc_ambient1.wav or "sox type5_noise_cirline_ofc_ambient1.wav -t wav - |" """)

    set_list = parse_set_parameter_strings(noise_set_para_array)

    pointsource_noise_list = []
    iso_noise_dict = {}
    for noise_set in set_list:
        current_noise_list = [noise_parser.parse_args(shlex.split(x.strip())) for x in open(noise_set.filename)]
        current_pointsource_noise_list = []
        for noise in current_noise_list:
            if sampling_rate is not None:
                # check if the rspecifier is a pipe or not
                if len(noise.noise_rspecifier.split()) == 1:
                    noise.noise_rspecifier = "sox {0} -r {1} -t wav - |".format(noise.noise_rspecifier, sampling_rate)
                else:
                    noise.noise_rspecifier = "{0} sox -t wav - -r {1} -t wav - |".format(noise.noise_rspecifier, sampling_rate)

            if noise.noise_type == "isotropic":
                if noise.room_linkage is None:
                    raise Exception("--room-linkage must be specified if --noise-type is isotropic")
                else:
                    if noise.room_linkage not in iso_noise_dict:
                        iso_noise_dict[noise.room_linkage] = []
                    iso_noise_dict[noise.room_linkage].append(noise)
            else:
                current_pointsource_noise_list.append(noise)

        pointsource_noise_list += smooth_probability_distribution(current_pointsource_noise_list, smoothing_weight, noise_set.probability)

    # ensure the point-source noise probabilities sum to 1
    pointsource_noise_list = smooth_probability_distribution(pointsource_noise_list, smoothing_weight, 1.0)
    if len(pointsource_noise_list) > 0:
        assert almost_equal(sum(noise.probability for noise in pointsource_noise_list), 1.0)

    # ensure the isotropic noise source probabilities for a given room sum to 1
    for key in iso_noise_dict.keys():
        iso_noise_dict[key] = smooth_probability_distribution(iso_noise_dict[key])
        assert almost_equal(sum(noise.probability for noise in iso_noise_dict[key]), 1.0)

    return (pointsource_noise_list, iso_noise_dict)


def main():
    args = get_args()

    random.seed(args.random_seed)
    rir_list = parse_rir_list(args.rir_set_para_array, args.rir_smoothing_weight, args.source_sampling_rate)
    print("Number of RIRs is {0}".format(len(rir_list)))
    pointsource_noise_list = []
    iso_noise_dict = {}
    if args.noise_set_para_array is not None:
        pointsource_noise_list, iso_noise_dict = parse_noise_list(args.noise_set_para_array,
                                                                args.noise_smoothing_weight,
                                                                args.source_sampling_rate)
        print("Number of point-source noises is {0}".format(len(pointsource_noise_list)))
        print("Number of isotropic noises is {0}".format(sum(len(iso_noise_dict[key]) for key in iso_noise_dict.keys())))
    room_dict = make_room_dict(rir_list)

    if args.include_original_data == "true":
        include_original = True
    else:
        include_original = False
    create_reverberated_copy(input_dir = args.input_dir,
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
                           max_noises_per_minute = args.max_noises_per_minute)


    data_lib.RunKaldiCommand("utils/validate_data_dir.sh --no-feats --no-text {output_dir}"
                    .format(output_dir = args.output_dir))

if __name__ == "__main__":
    main()

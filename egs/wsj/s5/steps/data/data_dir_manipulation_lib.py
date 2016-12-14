#!/usr/bin/env python
# Copyright 2016  Tom Ko
#           2016  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import subprocess, random, argparse, os, shlex, warnings

def RunKaldiCommand(command, wait = True):
    """ Runs commands frequently seen in Kaldi scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    #logger.info("Running the command\n{0}".format(command))
    p = subprocess.Popen(command, shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise Exception("There was an error while running the command {0}\n".format(command)+"-"*10+"\n"+stderr)
        return stdout, stderr
    else:
        return p

class list_cyclic_iterator:
  def __init__(self, list):
    self.list_index = 0
    self.list = list
    random.shuffle(self.list)

  def next(self):
    item = self.list[self.list_index]
    self.list_index = (self.list_index + 1) % len(self.list)
    return item

# This functions picks an item from the collection according to the associated probability distribution.
# The probability estimate of each item in the collection is stored in the "probability" field of
# the particular item. x : a collection (list or dictionary) where the values contain a field called probability
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

# This function smooths the probability distribution in the list
def SmoothProbabilityDistribution(list, smoothing_weight=0.0, target_sum=1.0):
    if len(list) > 0:
      num_unspecified = 0
      accumulated_prob = 0
      for item in list:
          if item.probability is None:
              num_unspecified += 1
          else:
              accumulated_prob += item.probability

      # Compute the probability for the items without specifying their probability
      uniform_probability = 0
      if num_unspecified > 0 and accumulated_prob < 1:
          uniform_probability = (1 - accumulated_prob) / float(num_unspecified)
      elif num_unspecified > 0 and accumulate_prob >= 1:
          warnings.warn("The sum of probabilities specified by user is larger than or equal to 1. "
                        "The items without probabilities specified will be given zero to their probabilities.")

      for item in list:
          if item.probability is None:
              item.probability = uniform_probability
          else:
              # smooth the probability
              item.probability = (1 - smoothing_weight) * item.probability + smoothing_weight * uniform_probability

      # Normalize the probability
      sum_p = sum(item.probability for item in list)
      for item in list:
          item.probability = item.probability / sum_p * target_sum

    return list

# This function parses a file and pack the data into a dictionary
# It is useful for parsing file like wav.scp, utt2spk, text...etc
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

# This function creates a file and write the content of a dictionary into it
def WriteDictToFile(dict, file_name):
    file = open(file_name, 'w')
    keys = dict.keys()
    keys.sort()
    for key in keys:
        value = dict[key]
        if type(value) in [list, tuple] :
            if type(value) is tuple:
                value = list(value)
            value.sort()
            value = ' '.join([ str(x) for x in value ])
        file.write('{0} {1}\n'.format(key, value))
    file.close()


# This function creates the utt2uniq file from the utterance id in utt2spk file
def CreateCorruptedUtt2uniq(input_dir, output_dir, num_replicas, include_original, prefix):
    corrupted_utt2uniq = {}
    # Parse the utt2spk to get the utterance id
    utt2spk = ParseFileToDict(input_dir + "/utt2spk", value_processor = lambda x: " ".join(x))
    keys = utt2spk.keys()
    keys.sort()
    if include_original:
        start_index = 0
    else:
        start_index = 1

    for i in range(start_index, num_replicas+1):
        for utt_id in keys:
            new_utt_id = GetNewId(utt_id, prefix, i)
            corrupted_utt2uniq[new_utt_id] = utt_id

    WriteDictToFile(corrupted_utt2uniq, output_dir + "/utt2uniq")

# This function generates a new id from the input id
# This is needed when we have to create multiple copies of the original data
# E.g. GetNewId("swb0035", prefix="rvb", copy=1) returns a string "rvb1_swb0035"
def GetNewId(id, prefix=None, copy=0):
    if prefix is not None:
        new_id = prefix + str(copy) + "_" + id
    else:
        new_id = id

    return new_id

# This function replicate the entries in files like segments, utt2spk, text
def AddPrefixToFields(input_file, output_file, num_replicas, include_original, prefix, field = [0]):
    list = map(lambda x: x.strip(), open(input_file))
    f = open(output_file, "w")
    if include_original:
        start_index = 0
    else:
        start_index = 1

    for i in range(start_index, num_replicas+1):
        for line in list:
            if len(line) > 0 and line[0] != ';':
                split1 = line.split()
                for j in field:
                    split1[j] = GetNewId(split1[j], prefix, i)
                print(" ".join(split1), file=f)
            else:
                print(line, file=f)
    f.close()

def CopyDataDirFiles(input_dir, output_dir, num_replicas, include_original, prefix):
    if not os.path.isfile(output_dir + "/wav.scp"):
        raise Exception("CopyDataDirFiles function expects output_dir to contain wav.scp already")

    AddPrefixToFields(input_dir + "/utt2spk", output_dir + "/utt2spk", num_replicas, include_original=include_original, prefix=prefix, field = [0,1])
    RunKaldiCommand("utils/utt2spk_to_spk2utt.pl <{output_dir}/utt2spk >{output_dir}/spk2utt"
                    .format(output_dir = output_dir))

    if os.path.isfile(input_dir + "/utt2uniq"):
        AddPrefixToFields(input_dir + "/utt2uniq", output_dir + "/utt2uniq", num_replicas, include_original=include_original, prefix=prefix, field =[0])
    else:
        # Create the utt2uniq file
        CreateCorruptedUtt2uniq(input_dir, output_dir, num_replicas, include_original, prefix)

    if os.path.isfile(input_dir + "/text"):
        AddPrefixToFields(input_dir + "/text", output_dir + "/text", num_replicas, include_original=include_original, prefix=prefix, field =[0])
    if os.path.isfile(input_dir + "/segments"):
        AddPrefixToFields(input_dir + "/segments", output_dir + "/segments", num_replicas, prefix=prefix, include_original=include_original, field = [0,1])
    if os.path.isfile(input_dir + "/reco2file_and_channel"):
        AddPrefixToFields(input_dir + "/reco2file_and_channel", output_dir + "/reco2file_and_channel", num_replicas, include_original=include_original, prefix=prefix, field = [0,1])

    AddPrefixToFields(input_dir + "/reco2dur", output_dir + "/reco2dur", num_replicas, include_original=include_original, prefix=prefix, field = [0])

    RunKaldiCommand("utils/validate_data_dir.sh --no-feats {output_dir}"
                    .format(output_dir = output_dir))


# This function parse the array of rir set parameter strings.
# It will assign probabilities to those rir sets which don't have a probability
# It will also check the existence of the rir list files.
def ParseSetParameterStrings(set_para_array):
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

    return SmoothProbabilityDistribution(set_list)


# This function creates the RIR list
# Each rir object in the list contains the following attributes:
# rir_id, room_id, receiver_position_id, source_position_id, rt60, drr, probability
# Please refer to the help messages in the parser for the meaning of these attributes
def ParseRirList(rir_set_para_array, smoothing_weight, sampling_rate = None):
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

    set_list = ParseSetParameterStrings(rir_set_para_array)

    rir_list = []
    for rir_set in set_list:
        current_rir_list = map(lambda x: rir_parser.parse_args(shlex.split(x.strip())),open(rir_set.filename))
        for rir in current_rir_list:
            if sampling_rate is not None:
                # check if the rspecifier is a pipe or not
                if len(rir.rir_rspecifier.split()) == 1:
                    rir.rir_rspecifier = "sox {0} -r {1} -t wav - |".format(rir.rir_rspecifier, sampling_rate)
                else:
                    rir.rir_rspecifier = "{0} sox -t wav - -r {1} -t wav - |".format(rir.rir_rspecifier, sampling_rate)

        rir_list += SmoothProbabilityDistribution(current_rir_list, smoothing_weight, rir_set.probability)

    return rir_list


# This dunction checks if the inputs are approximately equal assuming they are floats.
def almost_equal(value_1, value_2, accuracy = 10**-8):
    return abs(value_1 - value_2) < accuracy

# This function converts a list of RIRs into a dictionary of RIRs indexed by the room-id.
# Its values are objects with two attributes: a local RIR list
# and the probability of the corresponding room
# Please look at the comments at ParseRirList() for the attributes that a RIR object contains
def MakeRoomDict(rir_list):
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


# This function creates the point-source noise list
# and the isotropic noise dictionary from the noise information file
# The isotropic noise dictionary is indexed by the room
# and its value is the corrresponding isotropic noise list
# Each noise object in the list contains the following attributes:
# noise_id, noise_type, bg_fg_type, room_linkage, probability, noise_rspecifier
# Please refer to the help messages in the parser for the meaning of these attributes
def ParseNoiseList(noise_set_para_array, smoothing_weight, sampling_rate = None):
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

    set_list = ParseSetParameterStrings(noise_set_para_array)

    pointsource_noise_list = []
    iso_noise_dict = {}
    for noise_set in set_list:
        current_noise_list = map(lambda x: noise_parser.parse_args(shlex.split(x.strip())),open(noise_set.filename))
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

        pointsource_noise_list += SmoothProbabilityDistribution(current_pointsource_noise_list, smoothing_weight, noise_set.probability)

    # ensure the point-source noise probabilities sum to 1
    pointsource_noise_list = SmoothProbabilityDistribution(pointsource_noise_list, smoothing_weight, 1.0)
    if len(pointsource_noise_list) > 0:
        assert almost_equal(sum(noise.probability for noise in pointsource_noise_list), 1.0)

    # ensure the isotropic noise source probabilities for a given room sum to 1
    for key in iso_noise_dict.keys():
        iso_noise_dict[key] = SmoothProbabilityDistribution(iso_noise_dict[key])
        assert almost_equal(sum(noise.probability for noise in iso_noise_dict[key]), 1.0)

    return (pointsource_noise_list, iso_noise_dict)

def AddPointSourceNoise(room,  # the room selected
                        pointsource_noise_list, # the point source noise list
                        pointsource_noise_addition_probability, # Probability of adding point-source noises
                        foreground_snrs, # the SNR for adding the foreground noises
                        background_snrs, # the SNR for adding the background noises
                        speech_dur,  # duration of the recording
                        max_noises_recording,  # Maximum number of point-source noises that can be added
                        noise_addition_descriptor  # descriptor to store the information of the noise added
                        ):
    num_noises_added = 0
    if len(pointsource_noise_list) > 0 and random.random() < pointsource_noise_addition_probability and max_noises_recording >= 1:
        for k in range(random.randint(1, max_noises_recording)):
            num_noises_added = num_noises_added + 1
            # pick the RIR to reverberate the point-source noise
            noise = PickItemWithProbability(pointsource_noise_list)
            noise_rir = PickItemWithProbability(room.rir_list)
            # If it is a background noise, the noise will be extended and be added to the whole speech
            # if it is a foreground noise, the noise will not extended and be added at a random time of the speech
            if noise.bg_fg_type == "background":
                noise_rvb_command = """wav-reverberate --impulse-response="{0}" --duration={1}""".format(noise_rir.rir_rspecifier, speech_dur)
                noise_addition_descriptor['start_times'].append(0)
                noise_addition_descriptor['snrs'].append(background_snrs.next())
                noise_addition_descriptor['durations'].append(speech_dur)
                noise_addition_descriptor['noise_ids'].append(noise.noise_id)
            else:
                noise_rvb_command = """wav-reverberate --impulse-response="{0}" """.format(noise_rir.rir_rspecifier)
                noise_addition_descriptor['start_times'].append(round(random.random() * speech_dur, 2))
                noise_addition_descriptor['snrs'].append(foreground_snrs.next())
                noise_addition_descriptor['durations'].append(-1)
                noise_addition_descriptor['noise_ids'].append(noise.noise_id)

            # check if the rspecifier is a pipe or not
            if len(noise.noise_rspecifier.split()) == 1:
                noise_addition_descriptor['noise_io'].append("{1} {0} - |".format(noise.noise_rspecifier, noise_rvb_command))
            else:
                noise_addition_descriptor['noise_io'].append("{0} {1} - - |".format(noise.noise_rspecifier, noise_rvb_command))

# This function randomly decides whether to reverberate, and sample a RIR if it does
# It also decides whether to add the appropriate noises
# This function return the string of options to the binary wav-reverberate
def GenerateReverberationOpts(room_dict,  # the room dictionary, please refer to MakeRoomDict() for the format
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
    impulse_response_opts = ""
    additive_noise_opts = ""

    noise_addition_descriptor = {'noise_io': [],
                                 'start_times': [],
                                 'snrs': [],
                                 'noise_ids': [],
                                 'durations': []
                                 }
    # Randomly select the room
    # Here the room probability is a sum of the probabilities of the RIRs recorded in the room.
    room = PickItemWithProbability(room_dict)
    # Randomly select the RIR in the room
    speech_rir = PickItemWithProbability(room.rir_list)
    if random.random() < speech_rvb_probability:
        # pick the RIR to reverberate the speech
        impulse_response_opts = """--impulse-response="{0}" """.format(speech_rir.rir_rspecifier)

    rir_iso_noise_list = []
    if speech_rir.room_id in iso_noise_dict:
        rir_iso_noise_list = iso_noise_dict[speech_rir.room_id]
    # Add the corresponding isotropic noise associated with the selected RIR
    if len(rir_iso_noise_list) > 0 and random.random() < isotropic_noise_addition_probability:
        isotropic_noise = PickItemWithProbability(rir_iso_noise_list)
        # extend the isotropic noise to the length of the speech waveform
        # check if the rspecifier is really a pipe
        if len(isotropic_noise.noise_rspecifier.split()) == 1:
            noise_addition_descriptor['noise_io'].append("wav-reverberate --duration={1} {0} - |".format(isotropic_noise.noise_rspecifier, speech_dur))
        else:
            noise_addition_descriptor['noise_io'].append("{0} wav-reverberate --duration={1} - - |".format(isotropic_noise.noise_rspecifier, speech_dur))
        noise_addition_descriptor['start_times'].append(0)
        noise_addition_descriptor['snrs'].append(background_snrs.next())
        noise_addition_descriptor['noise_ids'].append(isotropic_noise.noise_id)
        noise_addition_descriptor['durations'].append(speech_dur)

    AddPointSourceNoise(room,  # the room selected
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

    return [impulse_response_opts, noise_addition_descriptor]


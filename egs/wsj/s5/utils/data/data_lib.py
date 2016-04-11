
import imp
import os

train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

def GetFrameShift(data_dir):
    frame_shift = train_lib.RunKaldiCommand("utils/data/get_frame_shift.sh {0}".format(data_dir))[0]
    return float(frame_shift.strip())

def GenerateUtt2Dur(data_dir):
    train_lib.RunKaldiCommand("utils/data/get_utt2dur.sh {0}".format(data_dir))

def GetUtt2Dur(data_dir):
    GenerateUtt2Dur(data_dir)
    utt2dur = {}
    for line in open('{0}/utt2dur'.format(data_dir), 'r').readlines():
        parts = line.split()
        utt2dur[parts[0]] = float(parts[1])
    return utt2dur

def GetUtt2Uniq(data_dir):
    utt2uniq_file = '{0}/utt2uniq'.format(data_dir)
    if not os.path.exists(utt2uniq_file):
        return None, None
    utt2uniq = {}
    uniq2utt = {}
    for line in open(utt2uniq_file, 'r').readlines():
        parts = line.split()
        utt2uniq[parts[0]] = parts[1]
        if uniq2utt.has_key(parts[1]):
            uniq2utt[parts[1]].append(parts[0])
        else:
            uniq2utt[parts[1]] = [parts[0]]
    return utt2uniq, uniq2utt

def GetNumFrames(data_dir, utts = None):
    GenerateUtt2Dur(data_dir)
    frame_shift = GetFrameShift(data_dir)
    total_duration = 0
    utt2dur = GetUtt2Dur(data_dir)
    if utts is None:
        utts = utt2dur.keys()
    for utt in utts:
        total_duration = total_duration + utt2dur[utt]
    return int(float(total_duration)/frame_shift)

def CreateDataLinks(file_names):
    train_lib.RunKaldiCommand(" utils/create_data_link.pl {0}".format(" ".join(file_names)))

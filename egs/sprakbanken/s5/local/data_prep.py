#!/usr/bin/env python
'''
# Copyright 2013-2014 Mirsk Digital Aps  (Author: Andreas Kirkedal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


This script outputs text1, wav.scp and utt2spk in the directory specified

'''


import sys
import os
import codecs
import locale

# set locale for sorting purposes

locale.setlocale(locale.LC_ALL, "C")

# global var


def unique(items):
    found = set([])
    keep = []

    for item in items:
        if item not in found:
            found.add(item)
            keep.append(item)

    return keep


def list2string(wordlist, lim=" ", newline=False):
    '''Converts a list to a string with a delimiter $lim and the possible
    addition of newline characters.'''
    strout = ""
    for w in wordlist:
        strout += w + lim
    if newline:
        return strout.strip(lim) + "\n"
    else:
        return strout.strip(lim)


def get_sprak_info(abspath):
    '''Returns the Sprakbank session id, utterance id and speaker id from the
    filename.'''
    return os.path.split(abspath)[-1].strip().split(".")[:-1]


def kaldi_utt_id(abspath, string=True):
    '''Creates the kaldi utterance id from the filename.'''
    fname = get_sprak_info(abspath)
    spkutt_id = [fname[1].strip(), fname[0].strip(), fname[2].strip()]
    if string:
        return list2string(spkutt_id, lim="-")
    else:
        return spkutt_id


def make_kaldi_text(line):
    '''Creates each line in the kaldi "text" file. '''
    txtfile = codecs.open(line.strip(), "r", "utf8").read()
    utt_id = kaldi_utt_id(line)
    return utt_id + " " + txtfile


def txt2wav(fstring):
    '''Changes the extension from .txt to .wav. '''
    return os.path.splitext(fstring)[0] + ".wav"


def make_kaldi_scp(line, sph, wav=False):
    if not wav:
        wavpath = txt2wav(line)
    else:
        wavpath = wav
    utt_id = kaldi_utt_id(line)
    return utt_id + " " + sph + " " + wavpath + " |\n"


def make_utt2spk(line):
    info = kaldi_utt_id(line, string=False)
    utt_id = list2string(info, lim="-")
    return list2string([utt_id, info[0]], newline=True)


def create_parallel_kaldi(filelist, sphpipe, snd=False):
    '''Creates the "text" file that maps a transcript to an utterance id and
    the corresponding wav.scp file. '''
    transcripts = []
    waves = []
    utt2spk = []
    for num, line in enumerate(filelist):
        transcriptline = make_kaldi_text(line)
        transcripts.append(transcriptline)
        if snd:
            scpline = make_kaldi_scp(line, sphpipe, snd[num].strip())
        else:
            scpline = make_kaldi_scp(line)
        waves.append(scpline)
        utt2spkline = make_utt2spk(line)
        utt2spk.append(utt2spkline)
    return (sorted(unique(transcripts)),
            sorted(unique(waves)),
            sorted(unique(utt2spk))
            )


if __name__ == '__main__':
    flist = codecs.open(sys.argv[1], "r").readlines()
    outpath = sys.argv[2]
    if len(sys.argv) == 5:
        sndlist = codecs.open(sys.argv[3], "r").readlines()
        sph2pipe = sys.argv[4] + " -f wav -p -c 1"
        traindata = create_parallel_kaldi(flist, sph2pipe, snd=sndlist)
    else:
        traindata = create_parallel_kaldi(flist, "")

    textout = codecs.open(os.path.join(outpath, "text.unnormalised"), "w", "utf8")
    wavout = codecs.open(os.path.join(outpath, "wav.scp"), "w")
    utt2spkout = codecs.open(os.path.join(outpath, "utt2spk"), "w")
    textout.writelines(traindata[0])
    wavout.writelines(traindata[1])
    utt2spkout.writelines(traindata[2])
    textout.close()
    wavout.close()
    utt2spkout.close()

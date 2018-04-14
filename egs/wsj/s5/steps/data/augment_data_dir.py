#!/usr/bin/env python3
# Copyright 2017  David Snyder
#           2017  Ye Bai
# Apache 2.0
#
# This script generates augmented data.  It is based on
# steps/data/reverberate_data_dir.py but doesn't handle reverberation.
# It is designed to be somewhat simpler and more flexible for augmenting with
# additive noise.
from __future__ import print_function
import sys, random, argparse, os, imp
sys.path.append("steps/data/")
from reverberate_data_dir import ParseFileToDict
from reverberate_data_dir import WriteDictToFile
data_lib = imp.load_source('dml', 'steps/data/data_dir_manipulation_lib.py')

def GetArgs():
    parser = argparse.ArgumentParser(description="Augment the data directory with additive noises. "
        "Noises are separated into background and foreground noises which are added together or "
        "separately.  Background noises are added to the entire recording, and repeated as necessary "
        "to cover the full length.  Multiple overlapping background noises can be added, to simulate "
        "babble, for example.  Foreground noises are added sequentially, according to a specified "
        "interval.  See also steps/data/reverberate_data_dir.py "
        "Usage: augment_data_dir.py [options...] <in-data-dir> <out-data-dir> "
        "E.g., steps/data/augment_data_dir.py --utt-suffix aug --fg-snrs 20:10:5:0 --bg-snrs 20:15:10 "
        "--num-bg-noise 1:2:3 --fg-interval 3 --fg-noise-dir data/musan_noise --bg-noise-dir "
        "data/musan_music data/train data/train_aug", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fg-snrs', type=str, dest = "fg_snr_str", default = '20:10:0',
                        help='When foreground noises are being added, the script will iterate through these SNRs.')
    parser.add_argument('--bg-snrs', type=str, dest = "bg_snr_str", default = '20:10:0',
                        help='When background noises are being added, the script will iterate through these SNRs.')
    parser.add_argument('--num-bg-noises', type=str, dest = "num_bg_noises", default = '1',
                        help='Number of overlapping background noises that we iterate over. For example, if the input is "1:2:3" then the output wavs will have either 1, 2, or 3 randomly chosen background noises overlapping the entire recording')
    parser.add_argument('--fg-interval', type=int, dest = "fg_interval", default = 0,
                        help='Number of seconds between the end of one foreground noise and the beginning of the next.')
    parser.add_argument('--utt-suffix', type=str, dest = "utt_suffix", default = "aug", help='Suffix added to utterance IDs.')
    parser.add_argument('--random-seed', type=int, dest = "random_seed", default = 123, help='Random seed.')

    parser.add_argument("--bg-noise-dir", type=str, dest="bg_noise_dir",
                        help="Background noise data directory")
    parser.add_argument("--fg-noise-dir", type=str, dest="fg_noise_dir",
                        help="Foreground noise data directory")
    parser.add_argument("input_dir", help="Input data directory")
    parser.add_argument("output_dir", help="Output data directory")

    print(' '.join(sys.argv))
    args = parser.parse_args()
    args = CheckArgs(args)
    return args

def CheckArgs(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.fg_interval >= 0:
        raise Exception("--fg-interval must be 0 or greater")
    if args.bg_noise_dir is None and args.fg_noise_dir is None:
        raise Exception("Either --fg-noise-dir or --bg-noise-dir must be specified")
    return args

def GetNoiseList(noise_wav_scp_filename):
    noise_wav_scp_file = open(noise_wav_scp_filename, 'r').readlines()
    noise_wavs = {}
    noise_utts = []
    for line in noise_wav_scp_file:
        toks=line.split(" ")
        wav = " ".join(toks[1:])
        noise_utts.append(toks[0])
        noise_wavs[toks[0]] = wav.rstrip()
    return noise_utts, noise_wavs

def AugmentWav(utt, wav, dur, fg_snr_opts, bg_snr_opts, fg_noise_utts, \
    bg_noise_utts, noise_wavs, noise2dur, interval, num_opts):
    # This section is common to both foreground and background noises
    new_wav = ""
    dur_str = str(dur)
    noise_dur = 0
    tot_noise_dur = 0
    snrs=[]
    noises=[]
    start_times=[]

    # Now handle the background noises
    if len(bg_noise_utts) > 0:
        num = random.choice(num_opts)
        for i in range(0, num):
            noise_utt = random.choice(bg_noise_utts)
            noise = "wav-reverberate --duration=" \
            + dur_str + " \"" + noise_wavs[noise_utt] + "\" - |"
            snr = random.choice(bg_snr_opts)
            snrs.append(snr)
            start_times.append(0)
            noises.append(noise)

    # Now handle the foreground noises
    if len(fg_noise_utts) > 0:
        while tot_noise_dur < dur:
            noise_utt = random.choice(fg_noise_utts)
            noise = noise_wavs[noise_utt]
            snr = random.choice(fg_snr_opts)
            snrs.append(snr)
            noise_dur = noise2dur[noise_utt]
            start_times.append(tot_noise_dur)
            tot_noise_dur += noise_dur + interval
            noises.append(noise)

    start_times_str = "--start-times='" + ",".join(map(str,start_times)) + "'"
    snrs_str = "--snrs='" + ",".join(map(str,snrs)) + "'"
    noises_str = "--additive-signals='" + ",".join(noises).strip() + "'"

    # If the wav is just a file
    if wav.strip()[-1] != "|":
        new_wav = "wav-reverberate --shift-output=true " + noises_str + " " \
            + start_times_str + " " + snrs_str + " " + wav + " - |"
    # Else if the wav is in a pipe
    else:
        new_wav = wav + " wav-reverberate --shift-output=true " + noises_str + " " \
            + start_times_str + " " + snrs_str + " - - |"
    return new_wav

def CopyFileIfExists(utt_suffix, filename, input_dir, output_dir):
    if os.path.isfile(input_dir + "/" + filename):
        dict = ParseFileToDict(input_dir + "/" + filename,
            value_processor = lambda x: " ".join(x))
        if len(utt_suffix) > 0:
            new_dict = {}
            for key in dict.keys():
                new_dict[key + "-" + utt_suffix] = dict[key]
            dict = new_dict
        WriteDictToFile(dict, output_dir + "/" + filename)

def main():
    args = GetArgs()
    fg_snrs = map(int, args.fg_snr_str.split(":"))
    bg_snrs = map(int, args.bg_snr_str.split(":"))
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_bg_noises = map(int, args.num_bg_noises.split(":"))
    reco2dur = ParseFileToDict(input_dir + "/reco2dur",
        value_processor = lambda x: float(x[0]))
    wav_scp_file = open(input_dir + "/wav.scp", 'r').readlines()

    noise_wavs = {}
    noise_reco2dur = {}
    bg_noise_utts = []
    fg_noise_utts = []

    # Load background noises
    if args.bg_noise_dir:
        bg_noise_wav_filename = args.bg_noise_dir + "/wav.scp"
        bg_noise_utts, bg_noise_wavs = GetNoiseList(bg_noise_wav_filename)
        bg_noise_reco2dur = ParseFileToDict(args.bg_noise_dir + "/reco2dur",
            value_processor = lambda x: float(x[0]))
        noise_wavs.update(bg_noise_wavs)
        noise_reco2dur.update(bg_noise_reco2dur)

    # Load background noises
    if args.fg_noise_dir:
        fg_noise_wav_filename = args.fg_noise_dir + "/wav.scp"
        fg_noise_reco2dur_filename = args.fg_noise_dir + "/reco2dur"
        fg_noise_utts, fg_noise_wavs = GetNoiseList(fg_noise_wav_filename)
        fg_noise_reco2dur = ParseFileToDict(args.fg_noise_dir + "/reco2dur",
            value_processor = lambda x: float(x[0]))
        noise_wavs.update(fg_noise_wavs)
        noise_reco2dur.update(fg_noise_reco2dur)

    random.seed(args.random_seed)
    new_utt2wav = {}
    new_utt2spk = {}

    # Augment each line in the wav file
    for line in wav_scp_file:
        toks = line.rstrip().split(" ")
        utt = toks[0]
        wav = " ".join(toks[1:])
        dur = reco2dur[utt]
        new_wav = AugmentWav(utt, wav, dur, fg_snrs, bg_snrs, fg_noise_utts,
            bg_noise_utts, noise_wavs, noise_reco2dur, args.fg_interval,
            num_bg_noises)
        new_utt = utt + "-" + args.utt_suffix
        new_utt2wav[new_utt] = new_wav

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    WriteDictToFile(new_utt2wav, output_dir + "/wav.scp")
    CopyFileIfExists(args.utt_suffix, "reco2dur", input_dir, output_dir)
    CopyFileIfExists(args.utt_suffix, "utt2dur", input_dir, output_dir)
    CopyFileIfExists(args.utt_suffix, "utt2spk", input_dir, output_dir)
    CopyFileIfExists(args.utt_suffix, "utt2lang", input_dir, output_dir)
    CopyFileIfExists(args.utt_suffix, "text", input_dir, output_dir)
    CopyFileIfExists(args.utt_suffix, "utt2spk", input_dir, output_dir)
    CopyFileIfExists(args.utt_suffix, "vad.scp", input_dir, output_dir)
    CopyFileIfExists("", "spk2gender", input_dir, output_dir)
    data_lib.RunKaldiCommand("utils/fix_data_dir.sh {output_dir}".format(output_dir = output_dir))

if __name__ == "__main__":
    main()

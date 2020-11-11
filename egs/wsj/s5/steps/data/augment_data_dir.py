#!/usr/bin/env python3
# Copyright 2017  David Snyder
#           2017  Ye Bai
#           2019  Phani Sankar Nidadavolu
# Apache 2.0
#
# This script generates augmented data.  It is based on
# steps/data/reverberate_data_dir.py but doesn't handle reverberation.
# It is designed to be somewhat simpler and more flexible for augmenting with
# additive noise.
from __future__ import print_function
import sys, random, argparse, os, imp
sys.path.append("steps/data/")
sys.path.insert(0, 'steps/')

from reverberate_data_dir import parse_file_to_dict
from reverberate_data_dir import write_dict_to_file
import libs.common as common_lib
data_lib = imp.load_source('dml', 'steps/data/data_dir_manipulation_lib.py')

def get_args():
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
    parser.add_argument('--num-bg-noises', type=str,
                        dest = "num_bg_noises", default = '1',
                        help='Number of overlapping background noises that we iterate over.'
                            ' For example, if the input is "1:2:3" then the output wavs will have either '
                            '1, 2, or 3 randomly chosen background noises overlapping the entire recording')
    parser.add_argument('--fg-interval', type=int,
                        dest = "fg_interval", default = 0,
                        help='Number of seconds between the end of one '
                            'foreground noise and the beginning of the next.')
    parser.add_argument('--utt-suffix', type=str,
                        dest = "utt_suffix", default = None,
                        help='Suffix added to utterance IDs.')
    parser.add_argument('--utt-prefix', type=str,
                        dest = "utt_prefix", default = None,
                        help='Prefix added to utterance IDs.')
    parser.add_argument('--random-seed', type=int, dest = "random_seed",
                        default = 123, help='Random seed.')
    parser.add_argument("--modify-spk-id", type=str,
                        dest='modify_spk_id', default=False,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"],
                        help='Utt prefix or suffix would be added to the spk id '
                            'also (used in ASR), in speaker id it is left unmodifed')
    parser.add_argument("--bg-noise-dir", type=str, dest="bg_noise_dir",
                        help="Background noise data directory")
    parser.add_argument("--fg-noise-dir", type=str, dest="fg_noise_dir",
                        help="Foreground noise data directory")
    parser.add_argument("input_dir", help="Input data directory")
    parser.add_argument("output_dir", help="Output data directory")

    print(' '.join(sys.argv))
    args = parser.parse_args()
    args = check_args(args)
    return args

def check_args(args):
    # Check args
    if args.utt_suffix is None and args.utt_prefix is None:
        args.utt_modifier_type = None
        args.utt_modifier = ""
    elif args.utt_suffix is None and args.utt_prefix is not None:
        args.utt_modifier_type = "prefix"
        args.utt_modifier = args.utt_prefix
    elif args.utt_suffix is not None and args.utt_prefix is None:
        args.utt_modifier_type = "suffix"
        args.utt_modifier = args.utt_suffix
    else:
        raise Exception("Trying to add both prefix and suffix. Choose either of them")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.fg_interval >= 0:
        raise Exception("--fg-interval must be 0 or greater")
    if args.bg_noise_dir is None and args.fg_noise_dir is None:
        raise Exception("Either --fg-noise-dir or --bg-noise-dir must be specified")
    return args

def get_noise_list(noise_wav_scp_filename):
    noise_wav_scp_file = open(noise_wav_scp_filename, 'r', encoding='utf-8').readlines()
    noise_wavs = {}
    noise_utts = []
    for line in noise_wav_scp_file:
        toks=line.split(" ")
        wav = " ".join(toks[1:])
        noise_utts.append(toks[0])
        noise_wavs[toks[0]] = wav.rstrip()
    return noise_utts, noise_wavs

def augment_wav(utt, wav, dur, fg_snr_opts, bg_snr_opts, fg_noise_utts, \
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

    start_times_str = "--start-times='" + ",".join([str(i) for i in start_times]) + "'"
    snrs_str = "--snrs='" + ",".join([str(i) for i in snrs]) + "'"
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

def get_new_id(utt, utt_modifier_type, utt_modifier):
    """ This function generates a new id from the input id
        This is needed when we have to create multiple copies of the original data
        E.g. get_new_id("swb0035", prefix="rvb", copy=1) returns a string "rvb1_swb0035"
    """
    if utt_modifier_type == "suffix" and len(utt_modifier) > 0:
        new_utt = utt + "-" + utt_modifier
    elif utt_modifier_type == "prefix" and len(utt_modifier) > 0:
        new_utt = utt_modifier + "-" + utt
    else:
        new_utt = utt

    return new_utt

def copy_file_if_exists(input_file, output_file, utt_modifier_type,
                        utt_modifier, fields=[0]):
    if os.path.isfile(input_file):
        clean_dict = parse_file_to_dict(input_file,
            value_processor = lambda x: " ".join(x))
        new_dict = {}
        for key in clean_dict.keys():
            modified_key = get_new_id(key, utt_modifier_type, utt_modifier)
            if len(fields) > 1:
                values = clean_dict[key].split(" ")
                modified_values = values
                for idx in range(1, len(fields)):
                    modified_values[idx-1] = get_new_id(values[idx-1],
                                            utt_modifier_type, utt_modifier)
                new_dict[modified_key] = " ".join(modified_values)
            else:
                new_dict[modified_key] = clean_dict[key]
        write_dict_to_file(new_dict, output_file)

def create_augmented_utt2uniq(input_dir, output_dir,
                            utt_modifier_type, utt_modifier):
    clean_utt2spk_file = input_dir + "/utt2spk"
    clean_utt2spk_dict = parse_file_to_dict(clean_utt2spk_file,
                            value_processor = lambda x: " ".join(x))
    augmented_utt2uniq_dict = {}
    for key in clean_utt2spk_dict.keys():
        modified_key = get_new_id(key, utt_modifier_type, utt_modifier)
        augmented_utt2uniq_dict[modified_key] = key
    write_dict_to_file(augmented_utt2uniq_dict, output_dir + "/utt2uniq")

def main():
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    fg_snrs = [int(i) for i in args.fg_snr_str.split(":")]
    bg_snrs = [int(i) for i in args.bg_snr_str.split(":")]
    num_bg_noises = [int(i) for i in args.num_bg_noises.split(":")]
    reco2dur = parse_file_to_dict(input_dir + "/reco2dur",
        value_processor = lambda x: float(x[0]))
    wav_scp_file = open(input_dir + "/wav.scp", 'r', encoding='utf-8').readlines()

    noise_wavs = {}
    noise_reco2dur = {}
    bg_noise_utts = []
    fg_noise_utts = []

    # Load background noises
    if args.bg_noise_dir:
        bg_noise_wav_filename = args.bg_noise_dir + "/wav.scp"
        bg_noise_utts, bg_noise_wavs = get_noise_list(bg_noise_wav_filename)
        bg_noise_reco2dur = parse_file_to_dict(args.bg_noise_dir + "/reco2dur",
            value_processor = lambda x: float(x[0]))
        noise_wavs.update(bg_noise_wavs)
        noise_reco2dur.update(bg_noise_reco2dur)

    # Load foreground noises
    if args.fg_noise_dir:
        fg_noise_wav_filename = args.fg_noise_dir + "/wav.scp"
        fg_noise_reco2dur_filename = args.fg_noise_dir + "/reco2dur"
        fg_noise_utts, fg_noise_wavs = get_noise_list(fg_noise_wav_filename)
        fg_noise_reco2dur = parse_file_to_dict(args.fg_noise_dir + "/reco2dur",
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
        new_wav = augment_wav(utt, wav, dur, fg_snrs, bg_snrs, fg_noise_utts,
            bg_noise_utts, noise_wavs, noise_reco2dur, args.fg_interval,
            num_bg_noises)

        new_utt = get_new_id(utt, args.utt_modifier_type, args.utt_modifier)

        new_utt2wav[new_utt] = new_wav

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_dict_to_file(new_utt2wav, output_dir + "/wav.scp")
    copy_file_if_exists(input_dir + "/reco2dur", output_dir + "/reco2dur",
                                args.utt_modifier_type, args.utt_modifier)
    copy_file_if_exists(input_dir + "/utt2dur", output_dir + "/utt2dur",
                                args.utt_modifier_type, args.utt_modifier)

    # Check whether to modify the speaker id or not while creating utt2spk file
    fields = ([0, 1] if args.modify_spk_id else [0])
    copy_file_if_exists(input_dir + "/utt2spk", output_dir + "/utt2spk",
                        args.utt_modifier_type, args.utt_modifier, fields=fields)
    copy_file_if_exists(input_dir + "/utt2lang", output_dir + "/utt2lang",
                        args.utt_modifier_type, args.utt_modifier)
    copy_file_if_exists(input_dir + "/utt2num_frames", output_dir + "/utt2num_frames",
                        args.utt_modifier_type, args.utt_modifier)
    copy_file_if_exists(input_dir + "/text", output_dir + "/text", args.utt_modifier_type,
                        args.utt_modifier)
    copy_file_if_exists(input_dir + "/segments", output_dir + "/segments",
                        args.utt_modifier_type, args.utt_modifier, fields=[0, 1])
    copy_file_if_exists(input_dir + "/vad.scp", output_dir + "/vad.scp",
                        args.utt_modifier_type, args.utt_modifier)
    copy_file_if_exists(input_dir + "/reco2file_and_channel",
                        output_dir + "/reco2file_and_channel",
                        args.utt_modifier_type, args.utt_modifier, fields=[0, 1])

    if args.modify_spk_id:
        copy_file_if_exists(input_dir + "/spk2gender", output_dir + "/spk2gender",
                        args.utt_modifier_type, args.utt_modifier)
    else:
        copy_file_if_exists(input_dir + "/spk2gender", output_dir + "/spk2gender", None, "")

    # Create utt2uniq file
    if os.path.isfile(input_dir + "/utt2uniq"):
        copy_file_if_exists(input_dir + "/utt2uniq", output_dir + "/utt2uniq",
                        args.utt_modifier_type, args.utt_modifier, fields=[0])
    else:
        create_augmented_utt2uniq(input_dir, output_dir,
                        args.utt_modifier_type, args.utt_modifier)

    data_lib.RunKaldiCommand("utils/utt2spk_to_spk2utt.pl <{output_dir}/utt2spk >{output_dir}/spk2utt"
                    .format(output_dir = output_dir))

    data_lib.RunKaldiCommand("utils/fix_data_dir.sh {output_dir}".format(output_dir = output_dir))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# This script is called by local/make_dihard_2018_dev.sh, and it creates the
# necessary files for DIHARD 2018 development directory.

import sys, os

def prepare_dihard_2018_dev(src_dir, data_dir):
    wavscp_fi = open(data_dir + "/wav.scp" , 'w')
    utt2spk_fi = open(data_dir + "/utt2spk" , 'w')
    segments_fi = open(data_dir + "/segments" , 'w')
    rttm_fi = open(data_dir + "/rttm" , 'w')
    reco2num_spk_fi = open(data_dir + "/reco2num_spk" , 'w')

    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            filename = os.path.join(subdir, file)
            if filename.endswith(".lab"):
                utt = os.path.basename(filename).split(".")[0]
                lines = open(filename, 'r').readlines()
                segment_id = 0
                for line in lines:
                    start, end, speech = line.split()
                    segment_id_str = "{}_{}".format(utt, str(segment_id).zfill(4))
                    segments_str = "{} {} {} {}\n".format(segment_id_str, utt, start, end)
                    utt2spk_str = "{} {}\n".format(segment_id_str, utt)
                    segments_fi.write(segments_str)
                    utt2spk_fi.write(utt2spk_str)
                    segment_id += 1
                wav_str = "{} sox -t flac {}/data/flac/{}.flac -t wav -r 16k "\
                        "-b 16 - channels 1 |\n".format(utt, src_dir, utt)
                wavscp_fi.write(wav_str)
                with open("{}/data/rttm/{}.rttm".format(src_dir, utt), 'r') as fh:
                    rttm_str = fh.read()
                rttm_fi.write(rttm_str)
                with open("{}/data/rttm/{}.rttm".format(src_dir, utt), 'r') as fh:
                    rttm_list = fh.readlines()
                spk_list = [(x.split())[7] for x in rttm_list] 
                num_spk = len(set(spk_list))
                reco2num_spk_fi.write("{} {}\n".format(utt, num_spk))
    wavscp_fi.close()
    utt2spk_fi.close()
    segments_fi.close()
    rttm_fi.close()
    reco2num_spk_fi.close()
    return 0

def main():
    src_dir = sys.argv[1]
    data_dir = sys.argv[2]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    prepare_dihard_2018_dev(src_dir, data_dir)
    return 0

if __name__=="__main__":
    main()

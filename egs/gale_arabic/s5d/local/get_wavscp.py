#!/usr/bin/env python3

import sys

relative = sys.argv[1]
audio_list = sys.argv[2]
wavscp_file = sys.argv[3]
kaldi_root = sys.argv[4]

def main():
    allowed_audio_type = ["wav", "flac", "sph"]
    data_dir =  "/".join(audio_list.split("/")[:-1])
    with open(audio_list, 'r') as al:
        with open(wavscp_file, 'w') as wf:
            for line in al.readlines():
                line = line.strip()
                audio_type = line.split(".")[-1]
                if audio_type not in allowed_audio_type:
                    print("Unrecognized audio file: {}, skip it".format(line))
                else:
                    utt_id = (line.split("/")[-1]).split(".")[0]
                    if not utt_id:
                        print("Invalid audio name: {}".format(line))
                    else:
                        prefix = data_dir + "/" if relative == "true" else ""
                        # wav and flac
                        if (audio_type == "wav" or audio_type == "flac"):
                            wf.write("{} sox {}{} -r 16000 -t wav - |\n".format(utt_id, prefix, line))
                        # sph
                        elif audio_type == "sph":
                            wf.write("{} {}/tools/sph2pipe_v2.5/sph2pipe -f wav -c 1 {}{} | sox - -r 16000 -t wav - |\n".format(utt_id, kaldi_root, prefix, line))

if __name__ == "__main__":
    main()

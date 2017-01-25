#! /usr/bin/env python

"""This script prints a mapping from phones to speech
activity labels
0 for silence, 1 for speech, 2 for noise and 3 for OOV.
Other labels can be optionally defined.
e.g. If 1, 2 and 3 are silence phones, 4, 5 and 6 are speech phones,
the SAD map would be
1 0
2 0
3 0
4 1
5 1
6 1.
The silence and speech are read from the phones/silence.txt and
phones/nonsilence.txt from the lang directory.
An initial SAD map can be provided using --init-sad-map to override
the above default mapping of phones. This is useful to say map
<UNK> or noise phones <NSN> to separate SAD labels.
"""

import argparse
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script prints a mapping from phones to speech
        activity labels
        0 for silence, 1 for speech, 2 for noise and 3 for OOV.
        Other labels can be optionally defined.
        e.g. If 1, 2 and 3 are silence phones, 4, 5 and 6 are speech phones,
        the SAD map would be
        1 0
        2 0
        3 0
        4 1
        5 1
        6 1.
        The silence and speech are read from the phones/silence.txt and
        phones/nonsilence.txt from the lang directory.
        An initial SAD map can be provided using --init-sad-map to override
        the above default mapping of phones. This is useful to say map
        <UNK> or noise phones <NSN> to separate SAD labels.
        """)

    parser.add_argument("--init-sad-map", type=str, action=common_lib.NullstrToNoneAction,
                        help="""Initial SAD map that will be used to override
                        the default mapping using phones/silence.txt and
                        phones/nonsilence.txt. Does not need to specify labels
                        for all the phones.
                        e.g.
                        <OOV> 3
                        <NSN> 2""")

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--noise-phones-file", type=str,
                             action=common_lib.NullstrToNoneAction,
                             help="Map noise phones from file to label 2")
    noise_group.add_argument("--noise-phones-list", type=str,
                             action=common_lib.NullstrToNoneAction,
                             help="A colon-separated list of noise phones to "
                             "map to label 2")
    parser.add_argument("--unk", type=str, action=common_lib.NullstrToNoneAction,
                        help="""UNK phone, if provided will be mapped to
                        label 3""")

    parser.add_argument("--map-noise-to-sil", type=str,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"], default=False,
                        help="""Map noise phones to silence before writing the
                        map. i.e. anything with label 2 is mapped to
                        label 0.""")
    parser.add_argument("--map-unk-to-speech", type=str,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"], default=False,
                        help="""Map UNK phone to speech before writing the map
                        i.e. anything with label 3 is mapped to label 1.""")

    parser.add_argument("lang_dir")

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    sad_map = {}

    for line in open('{0}/phones/nonsilence.txt'.format(args.lang_dir)):
        parts = line.strip().split()
        sad_map[parts[0]] = 1

    for line in open('{0}/phones/silence.txt'.format(args.lang_dir)):
        parts = line.strip().split()
        sad_map[parts[0]] = 0

    if args.init_sad_map is not None:
        for line in open(args.init_sad_map):
            parts = line.strip().split()
            try:
                sad_map[parts[0]] = int(parts[1])
            except Exception:
                raise Exception("Invalid line " + line)

    if args.unk is not None:
        sad_map[args.unk] = 3

    noise_phones = {}
    if args.noise_phones_file is not None:
        for line in open(args.noise_phones_file):
            parts = line.strip().split()
            noise_phones[parts[0]] = 1

    if args.noise_phones_list is not None:
        for x in args.noise_phones_list.split(":"):
            noise_phones[x] = 1

    for x, l in sad_map.iteritems():
        if l == 2 and args.map_noise_to_sil:
            l = 0
        if l == 3 and args.map_unk_to_speech:
            l = 1
        print ("{0} {1}".format(x, l))

if __name__ == "__main__":
    main()

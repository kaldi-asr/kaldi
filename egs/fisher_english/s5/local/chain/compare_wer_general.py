#! /usr/bin/env python

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        description="""
This script is used for comparing decoding results between systems.
e.g. local/chain/compare_wer_general.py exp/chain_cleaned/tdnn_{c,d}_sp
For use with discriminatively trained systems you specify the epochs after a colon:
for instance,
local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn_c_sp exp/chain_cleaned/tdnn_c_sp_smbr:{1,2,3}
""")

    parser.add_argument("--separator", type=str, default=" ",
                        help="Separator for different fields")
    parser.add_argument("--print-fine-details", action='store_true',
                        help="Add fine details of insertions, substitutions "
                        "and deletions.")
    parser.add_argument("--include-looped", action='store_true',
                        help="Used to include looped results")
    parser.add_argument("--field-size", type=int,
                        help="Field size for the models")
    parser.add_argument("systems", nargs='+')

    args = parser.parse_args()
    return args


def parse_system_string(system_string):
    parts = system_string.split(":")
    if len(parts) not in [1, 2, 3]:
        raise RuntimeError("Unable to parse system string {0}"
                           "".format(system_string))

    dir_name = parts[0]

    suffix = ""
    if len(parts) > 1:
        suffix = parts[1]

    model_name = os.path.basename(dir_name)
    if len(parts) > 2:
        model_name = parts[2]

    return (dir_name, suffix, model_name)


class SystemInfo(object):
    def __init__(self, dir_name, suffix, model_name):
        self.dir_name = dir_name
        self.suffix = suffix
        self.model_name = model_name
        self.iter_ = "final"

        if self.suffix != "":
            m = re.search("_iter(\d+)", suffix)
            if bool(m):
                self.iter_ = m.group(1)
        else:
            used_epochs = False

        self.probs = []
        self.wers = defaultdict(lambda: "NA")
        self.ins = defaultdict(lambda: "NA")
        self.dels = defaultdict(lambda: "NA")
        self.sub = defaultdict(lambda: "NA")

    def add_wer(self, dev_set, affix=""):
        decode_name = dev_set + self.suffix

        out = common_lib.get_command_stdout(
            "grep WER {dir_name}/decode{affix}_{decode_name}/wer* | utils/best_wer.sh"
            "".format(dir_name=self.dir_name, affix=affix,
                      decode_name=decode_name),
            require_zero_status=False)

        if out != "" and len(out.split()) >= 2:
            self.wers[(dev_set, affix)] = out.split()[1]
            self.ins[(dev_set, affix)] = out.split()[6]
            self.dels[(dev_set, affix)] = out.split()[8]
            self.sub[(dev_set, affix)] = out.split()[10]

    def _get_prob(self, set_="train", xent=False):

        if not os.path.exists(
            "{dir_name}/log/compute_prob_{set}.{iter}.log"
            "".format(dir_name=self.dir_name, set=set_, iter=self.iter_)):
            return "NA"

        out = common_lib.get_command_stdout(
            "grep Overall {dir_name}/log/compute_prob_{set}.{iter}.log | "
            "grep {opt} xent".format(dir_name=self.dir_name, set=set_,
                                     iter=self.iter_,
                                     opt="-w" if xent else "-v"),
            require_zero_status=False)

        if out == "":
            return "NA"

        lines = out.split("\n")
        prob = None

        affix = "-xent" if xent else ""
        for line in lines:
            if (bool(re.search(r"'output-0{0}'".format(affix), line))
                    or bool(re.search(r"'output{0}'".format(affix), line))):
                prob = float(line.split()[7])
                break

        return "NA" if prob is None else "{0:.4f}".format(prob)

    def add_probs(self):
        self.probs.append(self._get_prob(set_="train", xent=False))
        self.probs.append(self._get_prob(set_="valid", xent=False))
        self.probs.append(self._get_prob(set_="train", xent=True))
        self.probs.append(self._get_prob(set_="valid", xent=True))


def run(args):
    used_epochs = False
    systems = []
    for sys_string in args.systems:
        dir_name, suffix, model_name = parse_system_string(sys_string)
        info = SystemInfo(dir_name, suffix, model_name)

        if suffix != "" and re.search(suffix, "epoch"):
            used_epochs = True
        else:
            used_epochs = False

        for dev_set in ["dev", "test"]:
            info.add_wer(dev_set)

            if args.include_looped:
                info.add_wer(dev_set, affix="_looped")

        if not used_epochs:
            info.add_probs()

        systems.append(info)

    print_system_infos(args, systems, used_epochs)


def print_system_infos(args, system_infos, used_epochs=False):
    field_sizes = [args.field_size] * len(system_infos)

    if args.field_size is None:
        for i, x in enumerate(system_infos):
            field_sizes[i] = len(x.model_name)

    separator = args.separator
    print ("# {0: <25}{sep}{1}".format(
        "System",
        "{sep}".format(sep=args.separator).join(
            ["{0: <{1}}".format(x.model_name, field_sizes[i])
             for i, x in enumerate(system_infos)]),
        sep=args.separator))

    tups = set()
    for sys_info in system_infos:
        for tup in sys_info.wers:
            tups.add(tup)

    for tup in sorted(list(tups)):
        dev_set, affix = tup
        print ("# {0: <25}{sep}{1}".format(
            "WER on {0} {1}"
            "".format(dev_set, "[ "+affix+" ]" if affix != "" else ""),
            "{sep}".format(sep=args.separator).join(
                ["{0: <{1}}".format(x.wers[tup], field_sizes[i])
                 for i, x in enumerate(system_infos)]),
            sep=args.separator))
        if args.print_fine_details:
            print ("# {0: <25}{sep}{1}".format(
                "#Ins on {0} {1}"
                "".format(dev_set, "[ "+affix+" ]" if affix != "" else ""),
                "{sep}".format(sep=args.separator).join(
                    ["{0: <{1}}".format(x.ins[tup], field_sizes[i])
                     for i, x in enumerate(system_infos)]),
                sep=args.separator))
            print ("# {0: <25}{sep}{1}".format(
                "#Del on {0} {1}"
                "".format(dev_set, "[ "+affix+" ]" if affix != "" else ""),
                "{sep}".format(sep=args.separator).join(
                    ["{0: <{1}}".format(x.dels[tup], field_sizes[i])
                     for i, x in enumerate(system_infos)]),
                sep=args.separator))
            print ("# {0: <25}{sep}{1}".format(
                "#Sub on {0} {1}"
                "".format(dev_set, "[ "+affix+" ]" if affix != "" else ""),
                "{sep}".format(sep=args.separator).join(
                    ["{0: <{1}}".format(x.sub[tup], field_sizes[i])
                     for i, x in enumerate(system_infos)]),
                sep=args.separator))

    if not used_epochs:
        print ("# {0: <25}{sep}{1}".format(
            "Final train prob",
            "{sep}".format(sep=args.separator).join(
                ["{0: <{1}}".format(x.probs[0], field_sizes[i])
                 for i, x in enumerate(system_infos)]),
            sep=args.separator))

        print ("# {0: <25}{sep}{1}".format(
            "Final valid prob",
            "{sep}".format(sep=args.separator).join(
                ["{0: <{1}}".format(x.probs[1], field_sizes[i])
                 for i, x in enumerate(system_infos)]),
            sep=args.separator))

        print ("# {0: <25}{sep}{1}".format(
            "Final train prob (xent)",
            "{sep}".format(sep=args.separator).join(
                ["{0: <{1}}".format(x.probs[2], field_sizes[i])
                 for i, x in enumerate(system_infos)]),
            sep=args.separator))

        print ("# {0: <25}{sep}{1}".format(
            "Final valid prob (xent)",
            "{sep}".format(sep=args.separator).join(
                ["{0: <{1}}".format(x.probs[3], field_sizes[i])
                 for i, x in enumerate(system_infos)]),
            sep=args.separator))


if __name__ == "__main__":
    args = get_args()
    run(args)

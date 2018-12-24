#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.


# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
from __future__ import division
import sys
import re
import argparse

# expects the output of nnet3*train with --computation-debug=true
# will run faster if just the lines with "DebugAfterExecute" are provided
# <train-command> |grep DebugAfterExecute | steps/nnet3/report/summarize_compute_debug_timing.py

def GetArgs():
    parser = argparse.ArgumentParser(description="Summarizes the timing info from nnet3-*-train --computation.debug=true commands ")
    parser.add_argument("--node-prefixes", type=str,
                        help="list of prefixes. Execution times from nnet3 components with the same prefix"
                        " will be accumulated. Still distinguishes Propagate and BackPropagate commands"
                        " --node-prefixes Lstm1,Lstm2,Layer1", default=None)

    print(' '.join(sys.argv), file=sys.stderr)

    args = parser.parse_args()
    if args.node_prefixes is not None:
        raise NotImplementedError
        # this will be implemented after https://github.com/kaldi-asr/kaldi/issues/944
        args.node_prefixes = args.node_prefixes.split(',')
    else:
        args.node_prefixes = []

    return args
# get opening bracket position corresponding to the last closing bracket
def FindOpenParanthesisPosition(string):
    string = string.strip()
    if string[-1] != ")":
        # we don't know how to deal with these strings
        return None

    string_index = len(string) - 1
    closing_parans = []
    closing_parans.append(string_index)
    string_index -= 1
    while string_index >= 0:
        if string[string_index] == "(":
            if len(closing_parans) == 1:
                # this opening bracket corresponds to the last closing bracket
                return string_index
            else:
                closing_parans.pop()
        elif string[string_index] == ")":
            closing_parans.append(string_index)
        string_index -= 1

    raise Exception("Malformed string: Could not find opening paranthesis\n\t{0}".format(string))

# input : LOG (nnet3-chain-train:DebugAfterExecute():nnet-compute.cc:144) c68: BLstm1_backward_W_i-xr.Propagate(NULL, m6212(3136:3199, 0:555), &m31(0:63, 0:1023))
# output : BLstm1_backward_W_i-xr.Propagate
def ExtractCommandName(command_string):
    # create a concise representation for the the command
    # strip off : LOG (nnet3-chain-train:DebugAfterExecute():nnet-compute.cc:144)
    command = " ".join(command_string.split()[2:])
    # command = c68: BLstm1_backward_W_i-xr.Propagate(NULL, m6212(3136:3199, 0:555), &m31(0:63, 0:1023))
    end_position = FindOpenParanthesisPosition(command)
    if end_position is not None:
        command = command[:end_position]
    # command = c68: BLstm1_backward_W_i-xr.Propagate
    command = ":".join(command.split(":")[1:]).strip()
    # command = BLstm1_backward_W_i-xr.Propagate
    return command

def Main():
    # Sample Line
    # LOG (nnet3-chain-train:DebugAfterExecute():nnet-compute.cc:144) c128: m19 = []  |               |        time: 0.0007689 secs

    debug_regex = re.compile("DebugAfterExecute")
    command_times = {}
    for line in sys.stdin:
        parts = line.split("|")
        if len(parts) != 3:
            # we don't know how to deal with these lines
            continue
        if debug_regex.search(parts[0]) is not None:
            # this is a line printed in the DebugAfterExecute method

            # get the timing info
            time_parts = parts[-1].split()
            assert(len(time_parts) == 3 and time_parts[-1] == "secs" and time_parts[0] == "time:" )
            time = float(time_parts[1])

            command = ExtractCommandName(parts[0])
           # store the time
            try:
                command_times[command] += time
            except KeyError:
                command_times[command] = time

    total_time = sum(command_times.values())
    sorted_commands = sorted(command_times.items(), key = lambda x: x[1], reverse = True)
    for item in sorted_commands:
        print("{c} : time {t} : fraction {f}".format(c=item[0], t=item[1], f=float(item[1]) / total_time))


if __name__ == "__main__":
    args = GetArgs()
    Main()



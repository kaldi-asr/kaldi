#! /usr/bin/env python

from __future__ import print_function
import argparse, shlex

def GetArgs():
    parser = argparse.ArgumentParser(description="""This script generates a lang
directory for purpose of segmentation. It takes as arguments the list of phones,
the corresponding min durations and end transition probability.""")

    parser.add_argument("--phone-transition-parameters", dest='phone_transition_para_array',
                        type=str, action='append', required = True,
                        help = "Options to build topology. \n"
                        "--phone-list=<phone_list> # Colon-separated list of phones\n"
                        "--min-duration=<int>      # Min duration for the phones\n"
                        "--end-transition-probability=<float>   # Probability of the end transition after the minimum duration\n")
    parser.add_argument("dir", type=str,
                        help = "Output lang directory")
    args = parser.parse_args()
    return args


def ParsePhoneTransitionParameters(para_array):
    parser = argparse.ArgumentParser()

    parser.add_argument("--phone-list", type=str, required=True,
                        help="Colon-separated list of phones")
    parser.add_argument("--min-duration", type=int, default=3,
                        help="Minimum number of states for the phone")
    parser.add_argument("--end-transition-probability", type=float, default=0.1,
                        help="Probability of the end transition after the minimum duration")

    phone_transition_parameters = [ parser.parse_args(shlex.split(x)) for x in para_array ]

    for t in phone_transition_parameters:
        if (t.end_transition_probability > 1.0 or
            t.end_transition_probability < 0.0):
            raise ValueError("Expected --end-transition-probability to be "
                             "between 0 and 1, got {0} for phones {1}".format(
                                 t.end_transition_probability, t.phone_list))
        if t.min_duration > 100 or t.min_duration < 1:
            raise ValueError("Expected --min-duration to be "
                             "between 1 and 100, got {0} for phones {1}".format(
                                 t.min_duration, t.phone_list))

        t.phone_list = t.phone_list.split(":")

    return phone_transition_parameters

def GetPhoneMap(phone_transition_parameters):
    phone2int = {}
    n = 1
    for t in phone_transition_parameters:
        for p in t.phone_list:
            if p in phone2int:
                raise Exception("Phone {0} found in multiple topologies".format(p))
            phone2int[p] = n
            n += 1

    return phone2int

def Main():
    args = GetArgs()
    phone_transition_parameters = ParsePhoneTransitionParameters(args.phone_transition_para_array)

    phone2int = GetPhoneMap(phone_transition_parameters)

    topo = open("{0}/topo".format(args.dir), 'w')

    print ("<Topology>", file = topo)

    for t in phone_transition_parameters:
        print ("<TopologyEntry>", file = topo)
        print ("<ForPhones>", file = topo)
        print ("{0}".format(" ".join([str(phone2int[p]) for p in t.phone_list])), file = topo)
        print ("</ForPhones>", file = topo)

        for state in range(0, t.min_duration-1):
            print("<State> {0} <PdfClass> 0 <Transition> {1} 1.0 </State>".format(state, state + 1), file = topo)
        print("<State> {state} <PdfClass> 0 <Transition> {state} {self_prob} <Transition> {next_state} {next_prob} </State>".format(
            state = t.min_duration - 1, next_state = t.min_duration,
            self_prob = 1 - t.end_transition_probability,
            next_prob = t.end_transition_probability), file = topo)
        print("<State> {state} </State>".format(state = t.min_duration), file = topo) # Final state
        print ("</TopologyEntry>", file = topo)
    print ("</Topology>", file = topo)

    phones_file = open("{0}/phones.txt".format(args.dir), 'w')

    for p,n in sorted(list(phone2int.items()), key = lambda x:x[1]):
        print ("{0} {1}".format(p, n), file = phones_file)

if __name__ == '__main__':
    Main()

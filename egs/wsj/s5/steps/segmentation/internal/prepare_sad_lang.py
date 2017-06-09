#! /usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0

"""Prepares a graph directory with a simple HMM topology for segmentation.
"""

from __future__ import print_function
import argparse
import logging
import math
import os
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser(
        description="This script generates a graph directory for decoding with "
        "a simple HMM model.\n"
        "It needs as an input classes_info file with the format:\n"
        "<class-id (1-indexed)> <initial-probability> <self-loop-probability> "
        "<min-duration> <list-of-pairs>,\n"
        "where each pair is <destination-class>:<transition-probability>.\n"
        "destination-class -1 is used to represent final probabilitiy.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--transition-scale", type=float, default=1.0,
                        help="""Scale on transition probabilities relative to
                        LM weights""")
    parser.add_argument("--loopscale", type=float, default=0.1,
                        help="""Scale on self-loop log-probabilities relative
                        to LM weights""")
    parser.add_argument("classes_info", type=argparse.FileType('r'),
                        help="File with classes_info")
    parser.add_argument("dir", type=str,
                        help="Output lang directory")
    args = parser.parse_args()
    return args


class ClassInfo(object):
    def __init__(self, class_id):
        self.class_id = class_id
        self.start_state = -1   # start state for the class-id
        self.num_states = 0   # minimum duration constraint
        self.initial_prob = 0   # probability of transition into class-id
        self.self_loop_prob= 0
        # transition-probability indexed by destination class-id
        self.transitions = {}

    def __str__(self):
        return ("class-id={0},start-state={1},num-states={2},"
                "initial-prob={3:.2f},transitions={4}".format(
                    self.class_id, self.start_state, self.num_states,
                    self.initial_prob, ' '.join(
                        ['{0}:{1}'.format(x,y)
                         for x,y in self.transitions.iteritems()])))


def read_classes_info(file_handle):
    classes_info = {}

    num_states = 1
    num_classes = 0

    for line in file_handle.readlines():
        try:
            parts = line.split()
            class_id = int(parts[0])
            assert class_id > 0, class_id
            if class_id in classes_info:
                raise RuntimeError(
                    "Duplicate class-id {0} in file {1}".format(
                        class_id, file_handle.name))
            classes_info[class_id] = ClassInfo(class_id)
            class_info = classes_info[class_id]
            class_info.initial_prob = float(parts[1])
            class_info.self_loop_prob = float(parts[2])
            class_info.num_states = int(parts[3])
            class_info.start_state = num_states
            num_states += class_info.num_states
            num_classes += 1

            if len(parts) > 4:
                for part in parts[4:]:
                    dest_class, transition_prob = part.split(':')
                    dest_class = int(dest_class)
                    if dest_class in class_info.transitions:
                        logger.error(
                            "Duplicate transition to class-id {0}"
                            "in transitions".format(dest_class))
                        raise RuntimeError
                    class_info.transitions[dest_class] = float(transition_prob)
            else:
                raise RuntimeError(
                    "No transitions out of class {0}".format(class_id))
        except Exception:
            logger.error("Error processing line %s in file %s",
                         line, file_handle.name)
            raise
    logger.info("Got %d classes", num_classes)

    # Final state
    classes_info[-1] = ClassInfo(-1)
    class_info = classes_info[-1]
    class_info.num_states = 1
    class_info.start_state = num_states

    for class_id, class_info in classes_info.iteritems():
        logger.info("For class %d, got class-info %s", class_id, class_info)

    return classes_info


def print_states_for_class(args, class_id, classes_info, file_handle):
    class_info = classes_info[class_id]

    assert class_info.num_states >= 1, class_info

    # Print states for minimum duration constraint
    if class_info.num_states > 1:
        for state in range(class_info.start_state,
                           class_info.start_state + class_info.num_states - 1):
            print("{state} {dest_state} {class_id} {class_id} 0.0"
                  "".format(state=state, dest_state=state + 1,
                            class_id=class_id),
                  file=file_handle)

    state = class_info.start_state + class_info.num_states - 1

    transitions = []

    self_loop_cost = -args.loopscale * math.log(class_info.self_loop_prob)
    print("{state} {state} {class_id} {class_id} {cost}"
          "".format(state=state, class_id=class_id, cost=self_loop_cost),
          file=file_handle)
    forward_prob = 1.0 - class_info.self_loop_prob

    for dest_class, prob in class_info.transitions.iteritems():
        try:
            next_state = classes_info[dest_class].start_state

            print("{state} {next_state} {class_id} {class_id} "
                  "{cost}".format(
                      state=state, next_state=next_state, class_id=class_id,
                      cost=args.loopscale * math.log(forward_prob)
                      - args.transition_scale * math.log(prob / forward_prob)),
                  file=file_handle)
        except Exception:
            logger.error("Failed to add transition (%d->%d).\n"
                         "classes_info = %s", class_id, dest_class,
                         class_info)


def run(args):
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    classes_info = read_classes_info(args.classes_info)

    graph_file = open("{0}/HCLG.txt".format(args.dir), 'w')

    # Print transitions from initial state (initial probs)
    for class_id, class_info in classes_info.iteritems():
        if class_id == -1:
            continue
        class_info = classes_info[class_id]
        initial_cost = (-args.transition_scale
                        * math.log(class_info.initial_prob))
        print("0 {next_state} 0 0 {cost}"
              "".format(next_state=class_info.start_state,
                        cost=initial_cost),
              file=graph_file)

    for class_id, class_info in classes_info.iteritems():
        if class_id == -1:
            continue
        print_states_for_class(args, class_id, classes_info, graph_file)

    print("{state}".format(state=classes_info[-1].start_state),
          file=graph_file)
    graph_file.close()

    with open('{0}/phones.txt'.format(args.dir), 'w') as phones_f:
        print ("0 0", file=phones_f)
        n = 1
        for class_id, class_info in classes_info.iteritems():
            if class_id == -1:
                continue
            print ("{0} {1}".format(class_id, n), file=phones_f)
            n += 1

    common_lib.force_symlink('phones.txt'.format(args.dir),
                             '{0}/words.txt'.format(args.dir))


def main():
    try:
        args = get_args()
        run(args)
    except Exception:
        logger.error("Failed preparing lang directory")
        raise


if __name__ == '__main__':
    main()

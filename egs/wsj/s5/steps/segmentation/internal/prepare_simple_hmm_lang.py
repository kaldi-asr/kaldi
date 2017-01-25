#! /usr/bin/env python

from __future__ import print_function
import argparse
import logging
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
        description="""This script generates a lang directory for decoding with
        simple HMM model.
        It needs as an input classes_info file with the
        format:
        <class-id (1-indexed)> <initial-probability> <self-loop-probability> <min-duration> <list-of-pairs>,
        where each pair is <destination-class>:<transition-probability>.
        destination-class -1 is used to represent final probabilitiy.""")

    parser.add_argument("classes_info", type=argparse.FileType('r'),
                        help="File with classes_info")
    parser.add_argument("dir", type=str,
                        help="Output lang directory")
    args = parser.parse_args()
    return args


class ClassInfo(object):
    def __init__(self, class_id):
        self.class_id = class_id
        self.start_state = -1
        self.num_states = 0
        self.initial_prob = 0
        self.self_loop_prob= 0
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

    # Final state
    classes_info[-1] = ClassInfo(-1)
    class_info = classes_info[-1]
    class_info.num_states = 1
    class_info.start_state = num_states

    for class_id, class_info in classes_info.iteritems():
        logger.info("For class %d, dot class-info %s", class_id, class_info)

    return classes_info, num_classes


def print_states_for_class(class_id, classes_info, topo):
    class_info = classes_info[class_id]

    assert class_info.num_states > 1, class_info

    for state in range(class_info.start_state,
                       class_info.start_state + class_info.num_states - 1):
        print("<State> {state} <PdfClass> {pdf}"
              "<Transition> {dest_state} 1.0 </State>".format(
                  state=state, dest_state=state + 1,
                  pdf=class_info.class_id - 1),
              file=topo)

    state = class_info.start_state + class_info.num_states - 1

    transitions = []

    transitions.append("<Transition> {next_state} {next_prob}".format(
                       next_state=state, next_prob=class_info.self_loop_prob))

    for dest_class, prob in class_info.transitions.iteritems():
        try:
            next_state = classes_info[dest_class].start_state

            transitions.append("<Transition> {next_state} {next_prob}".format(
                                next_state=next_state, next_prob=prob))
        except Exception:
            logger.error("Failed to add transition (%d->%d).\n"
                         "classes_info = %s", class_id, dest_class,
                         class_info)

    print("<State> {state} <PdfClass> {pdf} "
          "{transitions} </State>".format(
              state=state, pdf=class_id - 1,
              transitions=' '.join(transitions)), file=topo)


def main():
    try:
        args = get_args()
        run(args)
    except Exception:
        logger.error("Failed preparing lang directory")
        raise


def run(args):
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    classes_info, num_classes = read_classes_info(args.classes_info)

    topo = open("{0}/topo".format(args.dir), 'w')

    print ("<Topology>", file=topo)
    print ("<TopologyEntry>", file=topo)
    print ("<ForPhones>", file=topo)
    print ("1", file=topo)
    print ("</ForPhones>", file=topo)

    # Print transitions from initial state (initial probs)
    transitions = []
    for class_id in range(1, num_classes + 1):
        class_info = classes_info[class_id]
        transitions.append("<Transition> {next_state} {next_prob}".format(
            next_state=class_info.start_state,
            next_prob=class_info.initial_prob))
    print("<State> 0 {transitions} </State>".format(
        transitions=' '.join(transitions)), file=topo)

    for class_id in range(1, num_classes + 1):
        print_states_for_class(class_id, classes_info, topo)

    print("<State> {state} </State>".format(
        state=classes_info[-1].start_state), file=topo)

    print ("</TopologyEntry>", file=topo)
    print ("</Topology>", file=topo)
    topo.close()

    with open('{0}/phones.txt'.format(args.dir), 'w') as phones_f:
        for class_id in range(1, num_classes + 1):
            print ("{0} {1}".format(class_id - 1, class_id), file=phones_f)

    common_lib.force_symlink('{0}/phones.txt'.format(args.dir),
                             '{0}/words.txt'.format(args.dir))


if __name__ == '__main__':
    main()

#! /usr/bin/env python

from __future__ import print_function
import argparse
import logging
import math


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
        description="""This script generates a bigram G.fst lang for decoding.
        It needs as an input classes_info file with the format:
        <class-id (1-indexed)> <initial-probability> <list-of-pairs>,
        where each pair is <destination-class>:<transition-probability>.
        destination-class -1 is used to represent final probabilitiy.""")

    parser.add_argument("classes_info", type=argparse.FileType('r'),
                        help="File with classes_info")
    parser.add_argument("out_file", type=argparse.FileType('w'),
                        help="Output G.fst. Use '-' for stdout")
    args = parser.parse_args()
    return args


class ClassInfo(object):
    def __init__(self, class_id):
        self.class_id = class_id
        self.start_state = -1
        self.initial_prob = 0
        self.transitions = {}

    def __str__(self):
        return ("class-id={0},start-state={1},"
                "initial-prob={2:.2f},transitions={3}".format(
                    self.class_id, self.start_state,
                    self.initial_prob, ' '.join(
                        ['{0}:{1}'.format(x, y)
                         for x, y in self.transitions.iteritems()])))


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
            class_info.start_state = num_states
            num_states += 1
            num_classes += 1

            total_prob = 0.0
            if len(parts) > 2:
                for part in parts[2:]:
                    dest_class, transition_prob = part.split(':')
                    dest_class = int(dest_class)
                    total_prob += float(transition_prob)

                    if total_prob > 1.0:
                        raise ValueError("total-probability out of class {0} "
                                         "is {1} > 1.0".format(class_id,
                                                               total_prob))

                    if dest_class in class_info.transitions:
                        logger.error(
                            "Duplicate transition to class-id {0}"
                            "in transitions".format(dest_class))
                        raise RuntimeError
                    class_info.transitions[dest_class] = float(transition_prob)

                if -1 in class_info.transitions:
                    if abs(total_prob - 1.0) > 0.001:
                        raise ValueError("total-probability out of class {0} "
                                         "is {1} != 1.0".format(class_id,
                                                                total_prob))
                else:
                    class_info.transitions[-1] = 1.0 - total_prob
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
    class_info.start_state = num_states

    for class_id, class_info in classes_info.iteritems():
        logger.info("For class %d, got class-info %s", class_id, class_info)

    return classes_info, num_classes


def print_states_for_class(class_id, classes_info, out_file):
    class_info = classes_info[class_id]

    state = class_info.start_state

    # Transition from the FST initial state
    print ("0 {end} <eps> <eps> {logprob}".format(
                end=state, logprob=-math.log(class_info.initial_prob)),
           file=out_file)

    for dest_class, prob in class_info.transitions.iteritems():
        try:
            if dest_class == class_id:  # self loop
                next_state = state
            else:  # other transition
                next_state = classes_info[dest_class].start_state

            print ("{start} {end} {class_id} {class_id} {logprob}".format(
                        start=state, end=next_state, class_id=class_id,
                        logprob=-math.log(prob)),
                   file=out_file)

        except Exception:
            logger.error("Failed to add transition (%d->%d).\n"
                         "classes_info = %s", class_id, dest_class,
                         class_info)

    print ("{start} {final} {class_id} {class_id}".format(
                start=state, final=classes_info[-1].start_state,
                class_id=class_id),
           file=out_file)
    print ("{0}".format(classes_info[-1].start_state), file=out_file)


def run(args):
    classes_info, num_classes = read_classes_info(args.classes_info)

    for class_id in range(1, num_classes + 1):
        print_states_for_class(class_id, classes_info, args.out_file)


def main():
    try:
        args = get_args()
        run(args)
    except Exception:
        logger.error("Failed to make G.fst")
        raise
    finally:
        for f in [args.classes_info, args.out_file]:
            if f is not None:
                f.close()


if __name__ == '__main__':
    main()

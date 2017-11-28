#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0

"""This module contains methods related to scheduling dropout.
See _self_test() for examples of how the functions work.
"""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_debug_dropout = False

def _parse_dropout_option(dropout_option):
    """Parses the string option to --trainer.dropout-schedule and
    returns a list of dropout schedules for different component name patterns.
    Calls _parse_dropout_string() function for each component name pattern
    in the option.

    Arguments:
        dropout_option: The string option passed to --trainer.dropout-schedule.
            See its help for details.
            See _self_test() for examples.
        num_archive_to_process: See _parse_dropout_string() for details.

    Returns a list of (component_name, dropout_schedule) tuples,
    where dropout_schedule is itself a list of
    (data_fraction, dropout_proportion) tuples sorted in reverse order of
    data_fraction.
    A data fraction of 0 corresponds to beginning of training
    and 1 corresponds to all data.
    """
    components = dropout_option.strip().split(' ')
    dropout_schedule = []
    for component in components:
        parts = component.split('=')

        if len(parts) == 2:
            component_name = parts[0]
            this_dropout_str = parts[1]
        elif len(parts) == 1:
            component_name = '*'
            this_dropout_str = parts[0]
        else:
            raise Exception("The dropout schedule must be specified in the "
                            "format 'pattern1=func1 patter2=func2' where "
                            "the pattern can be omitted for a global function "
                            "for all components.\n"
                            "Got {0} in {1}".format(component, dropout_option))

        this_dropout_values = _parse_dropout_string(this_dropout_str)
        dropout_schedule.append((component_name, this_dropout_values))

    if _debug_dropout:
        logger.info("Dropout schedules for component names is as follows:")
        logger.info("<component-name-pattern>: [(num_archives_processed), "
                    "(dropout_proportion) ...]")
        for name, schedule in dropout_schedule:
            logger.info("{0}: {1}".format(name, schedule))

    return dropout_schedule


def _parse_dropout_string(dropout_str):
    """Parses the dropout schedule from the string corresponding to a
    single component in --trainer.dropout-schedule.
    This is a module-internal function called by parse_dropout_function().

    Arguments:
        dropout_str: Specifies dropout schedule for a particular component
            name pattern.
            See help for the option --trainer.dropout-schedule.

    Returns a list of (data_fraction_processed, dropout_proportion) tuples
    sorted in descending order of num_archives_processed.
    A data fraction of 1 corresponds to all data.
    """
    dropout_values = []
    parts = dropout_str.strip().split(',')

    try:
        if len(parts) < 2:
            raise Exception("dropout proportion string must specify "
                            "at least the start and end dropouts")

        # Starting dropout proportion
        dropout_values.append((0, float(parts[0])))
        for i in range(1, len(parts) - 1):
            value_x_pair = parts[i].split('@')
            if len(value_x_pair) == 1:
                # Dropout proportion at half of training
                dropout_proportion = float(value_x_pair[0])
                data_fraction = 0.5
            else:
                assert len(value_x_pair) == 2

                dropout_proportion = float(value_x_pair[0])
                data_fraction = float(value_x_pair[1])

            if (data_fraction < dropout_values[-1][0]
                    or data_fraction > 1.0):
                logger.error(
                    "Failed while parsing value %s in dropout-schedule. "
                    "dropout-schedule must be in incresing "
                    "order of data fractions.", value_x_pair)
                raise ValueError

            dropout_values.append((data_fraction, float(dropout_proportion)))

        dropout_values.append((1.0, float(parts[-1])))
    except Exception:
        logger.error("Unable to parse dropout proportion string %s. "
                     "See help for option "
                     "--trainer.dropout-schedule.", dropout_str)
        raise

    # reverse sort so that its easy to retrieve the dropout proportion
    # for a particular data fraction
    dropout_values.reverse()
    for data_fraction, proportion in dropout_values:
        assert data_fraction <= 1.0 and data_fraction >= 0.0
        assert proportion <= 1.0 and proportion >= 0.0

    return dropout_values


def _get_component_dropout(dropout_schedule, data_fraction):
    """Retrieve dropout proportion from schedule when data_fraction
    proportion of data is seen. This value is obtained by using a
    piecewise linear function on the dropout schedule.
    This is a module-internal function called by _get_dropout_proportions().

    See help for --trainer.dropout-schedule for how the dropout value
    is obtained from the options.

    Arguments:
        dropout_schedule: A list of (data_fraction, dropout_proportion) values
            sorted in descending order of data_fraction.
        data_fraction: The fraction of data seen until this stage of
            training.
    """
    if data_fraction == 0:
        # Dropout at start of the iteration is in the last index of
        # dropout_schedule
        assert dropout_schedule[-1][0] == 0
        return dropout_schedule[-1][1]
    try:
        # Find lower bound of the data_fraction. This is the
        # lower end of the piecewise linear function.
        (dropout_schedule_index, initial_data_fraction,
         initial_dropout) = next((i, tup[0], tup[1])
                                 for i, tup in enumerate(dropout_schedule)
                                 if tup[0] <= data_fraction)
    except StopIteration:
        raise RuntimeError(
            "Could not find data_fraction in dropout schedule "
            "corresponding to data_fraction {0}.\n"
            "Maybe something wrong with the parsed "
            "dropout schedule {1}.".format(data_fraction, dropout_schedule))

    if dropout_schedule_index == 0:
        assert dropout_schedule[0][0] == 1 and data_fraction == 1
        return dropout_schedule[0][1]

    # The upper bound of data_fraction is at the index before the
    # lower bound.
    final_data_fraction, final_dropout = dropout_schedule[
        dropout_schedule_index - 1]

    if final_data_fraction == initial_data_fraction:
        assert data_fraction == initial_data_fraction
        return initial_dropout

    assert (data_fraction >= initial_data_fraction
            and data_fraction < final_data_fraction)

    return ((data_fraction - initial_data_fraction)
            * (final_dropout - initial_dropout)
            / (final_data_fraction - initial_data_fraction)
            + initial_dropout)


def _get_dropout_proportions(dropout_schedule, data_fraction):
    """Returns dropout proportions based on the dropout_schedule for the
    fraction of data seen at this stage of training.
    Returns None if dropout_schedule is None.

    Calls _get_component_dropout() for the different component name patterns
    in dropout_schedule.

    Arguments:
        dropout_schedule: Value for the --trainer.dropout-schedule option.
            See help for --trainer.dropout-schedule.
            See _self_test() for examples.
        data_fraction: The fraction of data seen until this stage of
            training.
    """
    if dropout_schedule is None:
        return None
    dropout_schedule = _parse_dropout_option(dropout_schedule)
    dropout_proportions = []
    for component_name, component_dropout_schedule in dropout_schedule:
        dropout_proportions.append(
            (component_name, _get_component_dropout(
                component_dropout_schedule, data_fraction)))
    return dropout_proportions


def get_dropout_edit_string(dropout_schedule, data_fraction, iter_):
    """Return an nnet3-copy --edits line to modify raw_model_string to
    set dropout proportions according to dropout_proportions.

    Arguments:
        dropout_schedule: Value for the --trainer.dropout-schedule option.
            See help for --trainer.dropout-schedule.
            See _self_test() for examples.

    See ReadEditConfig() in nnet3/nnet-utils.h to see how
    set-dropout-proportion directive works.
    """

    if dropout_schedule is None:
        return ""

    dropout_proportions = _get_dropout_proportions(
        dropout_schedule, data_fraction)

    edit_config_lines = []
    dropout_info = []

    for component_name, dropout_proportion in dropout_proportions:
        edit_config_lines.append(
            "set-dropout-proportion name={0} proportion={1}".format(
                component_name, dropout_proportion))
        dropout_info.append("pattern/dropout-proportion={0}/{1}".format(
            component_name, dropout_proportion))

    if _debug_dropout:
        logger.info("On iteration %d, %s", iter_, ', '.join(dropout_info))
    return ("""nnet3-copy --edits='{edits}' - - |""".format(
        edits=";".join(edit_config_lines)))


def _self_test():
    """Run self-test.
    This method is called if the module is run as a standalone script.
    """

    def assert_approx_equal(list1, list2):
        """Checks that the two dropout proportions lists are equal."""
        assert len(list1) == len(list2)
        for i in range(0, len(list1)):
            assert len(list1[i]) == 2
            assert len(list2[i]) == 2
            assert list1[i][0] == list2[i][0]
            assert abs(list1[i][1] - list2[i][1]) < 1e-8

    assert (_parse_dropout_option('*=0.0,0.5,0.0 lstm.*=0.0,0.3@0.75,0.0')
            == [ ('*', [ (1.0, 0.0), (0.5, 0.5), (0.0, 0.0) ]),
                 ('lstm.*', [ (1.0, 0.0), (0.75, 0.3), (0.0, 0.0) ]) ])
    assert_approx_equal(_get_dropout_proportions(
                           '*=0.0,0.5,0.0 lstm.*=0.0,0.3@0.75,0.0', 0.75),
                        [ ('*', 0.25), ('lstm.*', 0.3) ])
    assert_approx_equal(_get_dropout_proportions(
                            '*=0.0,0.5,0.0 lstm.*=0.0,0.3@0.75,0.0', 0.5),
                        [ ('*', 0.5), ('lstm.*', 0.2) ])
    assert_approx_equal(_get_dropout_proportions(
                            '*=0.0,0.5,0.0 lstm.*=0.0,0.3@0.75,0.0', 0.25),
                        [ ('*', 0.25), ('lstm.*', 0.1) ])

    assert (_parse_dropout_option('0.0,0.3,0.0')
            == [ ('*', [ (1.0, 0.0), (0.5, 0.3), (0.0, 0.0) ]) ])
    assert_approx_equal(_get_dropout_proportions('0.0,0.3,0.0', 0.5),
                        [ ('*', 0.3) ])
    assert_approx_equal(_get_dropout_proportions('0.0,0.3,0.0', 0.0),
                        [ ('*', 0.0) ])
    assert_approx_equal(_get_dropout_proportions('0.0,0.3,0.0', 1.0),
                        [ ('*', 0.0) ])
    assert_approx_equal(_get_dropout_proportions('0.0,0.3,0.0', 0.25),
                        [ ('*', 0.15) ])

    assert (_parse_dropout_option('0.0,0.5@0.25,0.0,0.6@0.75,0.0')
            == [ ('*', [ (1.0, 0.0), (0.75, 0.6), (0.5, 0.0), (0.25, 0.5), (0.0, 0.0) ]) ])
    assert_approx_equal(_get_dropout_proportions(
                            '0.0,0.5@0.25,0.0,0.6@0.75,0.0', 0.25),
                        [ ('*', 0.5) ])
    assert_approx_equal(_get_dropout_proportions(
                            '0.0,0.5@0.25,0.0,0.6@0.75,0.0', 0.1),
                        [ ('*', 0.2) ])

    assert (_parse_dropout_option('lstm.*=0.0,0.3,0.0@0.75,1.0')
            == [ ('lstm.*', [ (1.0, 1.0), (0.75, 0.0), (0.5, 0.3), (0.0, 0.0) ]) ])
    assert_approx_equal(_get_dropout_proportions(
                            'lstm.*=0.0,0.3,0.0@0.75,1.0', 0.25),
                        [ ('lstm.*', 0.15) ])
    assert_approx_equal(_get_dropout_proportions(
                            'lstm.*=0.0,0.3,0.0@0.75,1.0', 0.5),
                        [ ('lstm.*', 0.3) ])
    assert_approx_equal(_get_dropout_proportions(
                            'lstm.*=0.0,0.3,0.0@0.75,1.0', 0.9),
                        [ ('lstm.*', 0.6) ])


if __name__ == '__main__':
    try:
        _self_test()
    except Exception:
        logger.error("Failed self test")
        raise

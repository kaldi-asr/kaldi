

# Copyright 2016    Vijayaditya Peddinti
#                   Vimal Manohar
# Apache 2.0.

from __future__ import division
import datetime
import logging
import re

import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def parse_progress_logs_for_nonlinearity_stats(exp_dir):
    """ Parse progress logs for mean and std stats for non-linearities.

    e.g. for a line that is parsed from progress.*.log:
    exp/nnet3/lstm_self_repair_ld5_sp/log/progress.9.log:component name=Lstm3_i
    type=SigmoidComponent, dim=1280, self-repair-scale=1e-05, count=1.96e+05,
    value-avg=[percentiles(0,1,2,5 10,20,50,80,90
    95,98,99,100)=(0.05,0.09,0.11,0.15 0.19,0.27,0.50,0.72,0.83
    0.88,0.92,0.94,0.99), mean=0.502, stddev=0.23],
    deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90
    95,98,99,100)=(0.009,0.04,0.05,0.06 0.08,0.10,0.14,0.17,0.18
    0.19,0.20,0.20,0.21), mean=0.134, stddev=0.0397]
    """

    progress_log_files = "%s/log/progress.*.log" % (exp_dir)
    stats_per_component_per_iter = {}

    progress_log_lines = common_lib.run_kaldi_command(
        'grep -e "value-avg.*deriv-avg" {0}'.format(progress_log_files))[0]

    parse_regex = re.compile(
        ".*progress.([0-9]+).log:component name=(.+) "
        "type=(.*)Component,.*"
        "value-avg=\[.*mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*"
        "deriv-avg=\[.*mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\]")

    for line in progress_log_lines.split("\n"):
        mat_obj = parse_regex.search(line)
        if mat_obj is None:
            continue
        # groups = ('9', 'Lstm3_i', 'Sigmoid', '0.502', '0.23',
        # '0.134', '0.0397')
        groups = mat_obj.groups()
        iteration = int(groups[0])
        component_name = groups[1]
        component_type = groups[2]
        value_mean = float(groups[3])
        value_stddev = float(groups[4])
        deriv_mean = float(groups[5])
        deriv_stddev = float(groups[6])
        try:
            stats_per_component_per_iter[component_name][
                'stats'][iteration] = [value_mean, value_stddev,
                                       deriv_mean, deriv_stddev]
        except KeyError:
            stats_per_component_per_iter[component_name] = {}
            stats_per_component_per_iter[component_name][
                'type'] = component_type
            stats_per_component_per_iter[component_name]['stats'] = {}
            stats_per_component_per_iter[component_name][
                'stats'][iteration] = [value_mean, value_stddev,
                                       deriv_mean, deriv_stddev]

    return stats_per_component_per_iter


def parse_difference_string(string):
    dict = {}
    for parts in string.split():
        sub_parts = parts.split(":")
        dict[sub_parts[0]] = float(sub_parts[1])
    return dict


class MalformedClippedProportionLineException(Exception):
    def __init__(self, line):
        Exception.__init__(self,
                           "Malformed line encountered while trying to "
                           "extract clipped-proportions.\n{0}".format(line))


def parse_progress_logs_for_clipped_proportion(exp_dir):
    """ Parse progress logs for clipped proportion stats.

    e.g. for a line that is parsed from progress.*.log:
    exp/chain/cwrnn_trial2_ld5_sp/log/progress.245.log:component
    name=BLstm1_forward_c type=ClipGradientComponent, dim=512,
    norm-based-clipping=true, clipping-threshold=30,
    clipped-proportion=0.000565527,
    self-repair-clipped-proportion-threshold=0.01, self-repair-target=0,
    self-repair-scale=1
    """

    progress_log_files = "%s/log/progress.*.log" % (exp_dir)
    component_names = set([])
    progress_log_lines = common_lib.run_kaldi_command(
        'grep -e "{0}" {1}'.format(
            "clipped-proportion", progress_log_files))[0]
    parse_regex = re.compile(".*progress\.([0-9]+)\.log:component "
                             "name=(.*) type=.* "
                             "clipped-proportion=([0-9\.e\-]+)")

    cp_per_component_per_iter = {}

    max_iteration = 0
    component_names = set([])
    for line in progress_log_lines.split("\n"):
        mat_obj = parse_regex.search(line)
        if mat_obj is None:
            if line.strip() == "":
                continue
            raise MalformedClippedProportionLineException(line)
        groups = mat_obj.groups()
        iteration = int(groups[0])
        max_iteration = max(max_iteration, iteration)
        name = groups[1]
        clipped_proportion = float(groups[2])
        if clipped_proportion > 1:
            raise MalformedClippedProportionLineException(line)
        if iteration not in cp_per_component_per_iter:
            cp_per_component_per_iter[iteration] = {}
        cp_per_component_per_iter[iteration][name] = clipped_proportion
        component_names.add(name)
    component_names = list(component_names)
    component_names.sort()

    # re arranging the data into an array
    # and into an cp_per_iter_per_component
    cp_per_iter_per_component = {}
    for component_name in component_names:
        cp_per_iter_per_component[component_name] = []
    data = []
    data.append(["iteration"]+component_names)
    for iter in range(max_iteration+1):
        if iter not in cp_per_component_per_iter:
            continue
        comp_dict = cp_per_component_per_iter[iter]
        row = [iter]
        for component in component_names:
            try:
                row.append(comp_dict[component])
                cp_per_iter_per_component[component].append(
                    [iter, comp_dict[component]])
            except KeyError:
                # if clipped proportion is not available for a particular
                # component it is set to None
                # this usually happens during layer-wise discriminative
                # training
                row.append(None)
        data.append(row)

    return {'table': data,
            'cp_per_component_per_iter': cp_per_component_per_iter,
            'cp_per_iter_per_component': cp_per_iter_per_component}


def parse_progress_logs_for_param_diff(exp_dir, pattern):
    """ Parse progress logs for per-component parameter differences.

    e.g. for a line that is parsed from progress.*.log:
    exp/chain/cwrnn_trial2_ld5_sp/log/progress.245.log:LOG
    (nnet3-show-progress:main():nnet3-show-progress.cc:144) Relative parameter
    differences per layer are [ Cwrnn1_T3_W_r:0.0171537
    Cwrnn1_T3_W_x:1.33338e-07 Cwrnn1_T2_W_r:0.048075 Cwrnn1_T2_W_x:1.34088e-07
    Cwrnn1_T1_W_r:0.0157277 Cwrnn1_T1_W_x:0.0212704 Final_affine:0.0321521
    Cwrnn2_T3_W_r:0.0212082 Cwrnn2_T3_W_x:1.33691e-07 Cwrnn2_T2_W_r:0.0212978
    Cwrnn2_T2_W_x:1.33401e-07 Cwrnn2_T1_W_r:0.014976 Cwrnn2_T1_W_x:0.0233588
    Cwrnn3_T3_W_r:0.0237165 Cwrnn3_T3_W_x:1.33184e-07 Cwrnn3_T2_W_r:0.0239754
    Cwrnn3_T2_W_x:1.3296e-07 Cwrnn3_T1_W_r:0.0194809 Cwrnn3_T1_W_x:0.0271934 ]
    """

    if pattern not in set(["Relative parameter differences",
                           "Parameter differences"]):
        raise Exception("Unknown value for pattern : {0}".format(pattern))

    progress_log_files = "%s/log/progress.*.log" % (exp_dir)
    progress_per_iter = {}
    component_names = set([])
    progress_log_lines = common_lib.run_kaldi_command(
        'grep -e "{0}" {1}'.format(pattern, progress_log_files))[0]
    parse_regex = re.compile(".*progress\.([0-9]+)\.log:"
                             "LOG.*{0}.*\[(.*)\]".format(pattern))
    for line in progress_log_lines.split("\n"):
        mat_obj = parse_regex.search(line)
        if mat_obj is None:
            continue
        groups = mat_obj.groups()
        iteration = groups[0]
        differences = parse_difference_string(groups[1])
        component_names = component_names.union(differences.keys())
        progress_per_iter[int(iteration)] = differences

    component_names = list(component_names)
    component_names.sort()
    # rearranging the parameter differences available per iter
    # into parameter differences per component
    progress_per_component = {}
    for cn in component_names:
        progress_per_component[cn] = {}

    max_iter = max(progress_per_iter.keys())
    total_missing_iterations = 0
    gave_user_warning = False
    for iter in range(max_iter + 1):
        try:
            component_dict = progress_per_iter[iter]
        except KeyError:
            continue

        for component_name in component_names:
            try:
                progress_per_component[component_name][iter] = component_dict[
                    component_name]
            except KeyError:
                total_missing_iterations += 1
                # the component was not found this iteration, may be because of
                # layerwise discriminative training
                pass
        if (total_missing_iterations/len(component_names) > 20
                and not gave_user_warning and logger is not None):
            logger.warning("There are more than {0} missing iterations per "
                           "component. Something might be wrong.".format(
                                total_missing_iterations/len(component_names)))
            gave_user_warning = True

    return {'progress_per_component': progress_per_component,
            'component_names': component_names,
            'max_iter': max_iter}


def parse_train_logs(exp_dir):
    train_log_files = "%s/log/train.*.log" % (exp_dir)
    train_log_lines = common_lib.run_kaldi_command(
        'grep -e Accounting {0}'.format(train_log_files))[0]
    parse_regex = re.compile(".*train\.([0-9]+)\.([0-9]+)\.log:# "
                             "Accounting: time=([0-9]+) thread.*")

    train_times = {}
    for line in train_log_lines.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            try:
                train_times[int(groups[0])][int(groups[1])] = float(groups[2])
            except KeyError:
                train_times[int(groups[0])] = {}
                train_times[int(groups[0])][int(groups[1])] = float(groups[2])
    iters = train_times.keys()
    for iter in iters:
        values = train_times[iter].values()
        train_times[iter] = max(values)
    return train_times


def parse_prob_logs(exp_dir, key='accuracy', output="output"):
    train_prob_files = "%s/log/compute_prob_train.*.log" % (exp_dir)
    valid_prob_files = "%s/log/compute_prob_valid.*.log" % (exp_dir)
    train_prob_strings = common_lib.run_kaldi_command(
        'grep -e {0} {1}'.format(key, train_prob_files), wait=True)[0]
    valid_prob_strings = common_lib.run_kaldi_command(
        'grep -e {0} {1}'.format(key, valid_prob_files))[0]

    # LOG
    # (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:149)
    # Overall log-probability for 'output' is -0.399395 + -0.013437 = -0.412832
    # per frame, over 20000 fra

    # LOG
    # (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:144)
    # Overall log-probability for 'output' is -0.307255 per frame, over 20000
    # frames.

    parse_regex = re.compile(
        ".*compute_prob_.*\.([0-9]+).log:LOG "
        ".nnet3.*compute-prob:PrintTotalStats..:"
        "nnet.*diagnostics.cc:[0-9]+. Overall ([a-zA-Z\-]+) for "
        "'{output}'.*is ([0-9.\-e]+) .*per frame".format(output=output))

    train_loss = {}
    valid_loss = {}

    for line in train_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                train_loss[int(groups[0])] = groups[2]
    for line in valid_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                valid_loss[int(groups[0])] = groups[2]
    iters = list(set(valid_loss.keys()).intersection(train_loss.keys()))
    iters.sort()
    return map(lambda x: (int(x), float(train_loss[x]),
                          float(valid_loss[x])), iters)


def generate_accuracy_report(exp_dir, key="accuracy", output="output"):
    times = parse_train_logs(exp_dir)
    data = parse_prob_logs(exp_dir, key, output)
    report = []
    report.append("%Iter\tduration\ttrain_loss\tvalid_loss\tdifference")
    for x in data:
        try:
            report.append("%d\t%s\t%g\t%g\t%g" % (x[0], str(times[x[0]]),
                                                  x[1], x[2], x[2]-x[1]))
        except KeyError:
            continue

    total_time = 0
    for iter in times.keys():
        total_time += times[iter]
    report.append("Total training time is {0}\n".format(
                    str(datetime.timedelta(seconds=total_time))))
    return ["\n".join(report), times, data]

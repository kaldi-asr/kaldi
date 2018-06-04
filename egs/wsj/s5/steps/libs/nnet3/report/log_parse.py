

# Copyright 2016    Vijayaditya Peddinti
#                   Vimal Manohar
# Apache 2.0.

from __future__ import division
from __future__ import print_function
import traceback
import datetime
import logging
import re

import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

g_lstmp_nonlin_regex_pattern = ''.join([".*progress.([0-9]+).log:component name=(.+) ",
    "type=(.*)Component,.*",
    "i_t_sigmoid.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "f_t_sigmoid.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "c_t_tanh.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "o_t_sigmoid.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "m_t_tanh.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\]"])

g_normal_nonlin_regex_pattern = ''.join([".*progress.([0-9]+).log:component name=(.+) ",
    "type=(.*)Component,.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\]"])

g_normal_nonlin_regex_pattern_with_oderiv = ''.join([".*progress.([0-9]+).log:component name=(.+) ",
    "type=(.*)Component,.*",
    "value-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "deriv-avg=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*",
    "oderiv-rms=\[.*=\((.+)\), mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\]"])

class KaldiLogParseException(Exception):
    """ An Exception class that throws an error when there is an issue in
    parsing the log files. Extend this class if more granularity is needed.
    """
    def __init__(self, message = None):
        if message is not None and message.strip() == "":
            message = None

        Exception.__init__(self,
                           "There was an error while trying to parse the logs."
                           " Details : \n{0}\n".format(message))

# This function is used to fill stats_per_component_per_iter table with the
# results of regular expression.

def fill_nonlin_stats_table_with_regex_result(groups, gate_index, stats_table):
    iteration = int(groups[0])
    component_name = groups[1]
    component_type = groups[2]
    # for value-avg
    value_percentiles = groups[3+gate_index*6]
    value_mean = float(groups[4+gate_index*6])
    value_stddev = float(groups[5+gate_index*6])
    value_percentiles_split = re.split(',| ',value_percentiles)
    assert len(value_percentiles_split) == 13
    value_5th = float(value_percentiles_split[4])
    value_50th = float(value_percentiles_split[6])
    value_95th = float(value_percentiles_split[9])
    # for deriv-avg
    deriv_percentiles = groups[6+gate_index*6]
    deriv_mean = float(groups[7+gate_index*6])
    deriv_stddev = float(groups[8+gate_index*6])
    deriv_percentiles_split = re.split(',| ',deriv_percentiles)
    assert len(deriv_percentiles_split) == 13
    deriv_5th = float(deriv_percentiles_split[4])
    deriv_50th = float(deriv_percentiles_split[6])
    deriv_95th = float(deriv_percentiles_split[9])

    if len(groups) <= 9:
        try:
            if stats_table[component_name]['stats'].has_key(iteration):
                stats_table[component_name]['stats'][iteration].extend(
                        [value_mean,  value_stddev,
                         deriv_mean,  deriv_stddev,
                         value_5th,  value_50th,  value_95th,
                         deriv_5th,  deriv_50th,  deriv_95th])
            else:
                stats_table[component_name]['stats'][iteration] = [
                        value_mean,  value_stddev,
                        deriv_mean,  deriv_stddev,
                        value_5th,  value_50th,  value_95th,
                        deriv_5th,  deriv_50th,  deriv_95th]
        except KeyError:
            stats_table[component_name] = {}
            stats_table[component_name]['type'] = component_type
            stats_table[component_name]['stats'] = {}
            stats_table[component_name][
                    'stats'][iteration] = [value_mean,  value_stddev,
                                           deriv_mean,  deriv_stddev,
                                           value_5th,  value_50th,  value_95th,
                                           deriv_5th,  deriv_50th,  deriv_95th]
    else:
        #for oderiv-rms
        oderiv_percentiles = groups[9+gate_index*6]
        oderiv_mean = float(groups[10+gate_index*6])
        oderiv_stddev = float(groups[11+gate_index*6])
        oderiv_percentiles_split = re.split(',| ',oderiv_percentiles)
        assert len(oderiv_percentiles_split) == 13
        oderiv_5th = float(oderiv_percentiles_split[4])
        oderiv_50th = float(oderiv_percentiles_split[6])
        oderiv_95th = float(oderiv_percentiles_split[9])
        try:
            if stats_table[component_name]['stats'].has_key(iteration):
                stats_table[component_name]['stats'][iteration].extend(
                        [value_mean,  value_stddev,
                         deriv_mean,  deriv_stddev,
                         oderiv_mean, oderiv_stddev,
                         value_5th,  value_50th,  value_95th,
                         deriv_5th,  deriv_50th,  deriv_95th,
                         oderiv_5th, oderiv_50th, oderiv_95th])
            else:
                stats_table[component_name]['stats'][iteration] = [
                        value_mean,  value_stddev,
                        deriv_mean,  deriv_stddev,
                        oderiv_mean, oderiv_stddev,
                        value_5th,  value_50th,  value_95th,
                        deriv_5th,  deriv_50th,  deriv_95th,
                        oderiv_5th, oderiv_50th, oderiv_95th]
        except KeyError:
            stats_table[component_name] = {}
            stats_table[component_name]['type'] = component_type
            stats_table[component_name]['stats'] = {}
            stats_table[component_name][
                    'stats'][iteration] = [value_mean,  value_stddev,
                                           deriv_mean,  deriv_stddev,
                                           oderiv_mean, oderiv_stddev,
                                           value_5th,  value_50th,  value_95th,
                                           deriv_5th,  deriv_50th,  deriv_95th,
                                           oderiv_5th, oderiv_50th, oderiv_95th]

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

    progress_log_lines = common_lib.get_command_stdout(
        'grep -e "value-avg.*deriv-avg.*oderiv" {0}'.format(progress_log_files),
        require_zero_status = False)

    if progress_log_lines:
        # cases with oderiv-rms
        parse_regex = re.compile(g_normal_nonlin_regex_pattern_with_oderiv)
    else:
        # cases with only value-avg and deriv-avg
        progress_log_lines = common_lib.get_command_stdout(
        'grep -e "value-avg.*deriv-avg" {0}'.format(progress_log_files),
        require_zero_status = False)
        parse_regex = re.compile(g_normal_nonlin_regex_pattern)

    for line in progress_log_lines.split("\n"):
        mat_obj = parse_regex.search(line)
        if mat_obj is None:
            continue
        # groups = ('9', 'Lstm3_i', 'Sigmoid', '0.05...0.99', '0.502', '0.23',
        # '0.009...0.21', '0.134', '0.0397')
        groups = mat_obj.groups()
        component_type = groups[2]
        if component_type == 'LstmNonlinearity':
            parse_regex_lstmp = re.compile(g_lstmp_nonlin_regex_pattern)
            mat_obj = parse_regex_lstmp.search(line)
            groups = mat_obj.groups()
            assert len(groups) == 33
            for i in list(range(0,5)):
                fill_nonlin_stats_table_with_regex_result(groups, i,
                        stats_per_component_per_iter)
        else:
            fill_nonlin_stats_table_with_regex_result(groups, 0,
                    stats_per_component_per_iter)
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
    progress_log_lines = common_lib.get_command_stdout(
        'grep -e "{0}" {1}'.format(
            "clipped-proportion", progress_log_files),
        require_zero_status=False)
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
    progress_log_lines = common_lib.get_command_stdout(
        'grep -e "{0}" {1}'.format(pattern, progress_log_files))
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


def get_train_times(exp_dir):
    train_log_files = "%s/log/" % (exp_dir)
    train_log_names = "train.*.log"
    train_log_lines = common_lib.get_command_stdout(
        'find {0} -name "{1}" | xargs grep -H -e Accounting'.format(train_log_files,train_log_names))
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
    train_prob_strings = common_lib.get_command_stdout(
        'grep -e {0} {1}'.format(key, train_prob_files))
    valid_prob_strings = common_lib.get_command_stdout(
        'grep -e {0} {1}'.format(key, valid_prob_files))

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
        ".nnet3.*compute-prob.*:PrintTotalStats..:"
        "nnet.*diagnostics.cc:[0-9]+. Overall ([a-zA-Z\-]+) for "
        "'{output}'.*is ([0-9.\-e]+) .*per frame".format(output=output))

    train_objf = {}
    valid_objf = {}

    for line in train_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                train_objf[int(groups[0])] = groups[2]
    if not train_objf:
        raise KaldiLogParseException("Could not find any lines with {k} in "
                " {l}".format(k=key, l=train_prob_files))

    for line in valid_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                valid_objf[int(groups[0])] = groups[2]

    if not valid_objf:
        raise KaldiLogParseException("Could not find any lines with {k} in "
                " {l}".format(k=key, l=valid_prob_files))

    iters = list(set(valid_objf.keys()).intersection(train_objf.keys()))
    if not iters:
        raise KaldiLogParseException("Could not any common iterations with"
                " key {k} in both {tl} and {vl}".format(
                    k=key, tl=train_prob_files, vl=valid_prob_files))
    iters.sort()
    return list(map(lambda x: (int(x), float(train_objf[x]),
                               float(valid_objf[x])), iters))

def parse_rnnlm_prob_logs(exp_dir, key='objf'):
    train_prob_files = "%s/log/train.*.*.log" % (exp_dir)
    valid_prob_files = "%s/log/compute_prob.*.log" % (exp_dir)
    train_prob_strings = common_lib.get_command_stdout(
        'grep -e {0} {1}'.format(key, train_prob_files))
    valid_prob_strings = common_lib.get_command_stdout(
        'grep -e {0} {1}'.format(key, valid_prob_files))

    # LOG
    # (rnnlm-train[5.3.36~8-2ec51]:PrintStatsOverall():rnnlm-core-training.cc:118)
    # Overall objf is (-4.426 + -0.008287) = -4.435 over 4.503e+06 words (weighted)
    # in 1117 minibatches; exact = (-4.426 + 0) = -4.426

    # LOG
    # (rnnlm-compute-prob[5.3.36~8-2ec51]:PrintStatsOverall():rnnlm-core-training.cc:118)
    # Overall objf is (-4.677 + -0.002067) = -4.679 over 1.08e+05 words (weighted)
    # in 27 minibatches; exact = (-4.677 + 0.002667) = -4.674

    parse_regex_train = re.compile(
        ".*train\.([0-9]+).1.log:LOG "
        ".rnnlm-train.*:PrintStatsOverall..:"
        "rnnlm.*training.cc:[0-9]+. Overall ([a-zA-Z\-]+) is "
        ".*exact = \(.+\) = ([0-9.\-\+e]+)")

    parse_regex_valid = re.compile(
        ".*compute_prob\.([0-9]+).log:LOG "
        ".rnnlm.*compute-prob.*:PrintStatsOverall..:"
        "rnnlm.*training.cc:[0-9]+. Overall ([a-zA-Z\-]+) is "
        ".*exact = \(.+\) = ([0-9.\-\+e]+)")

    train_objf = {}
    valid_objf = {}

    for line in train_prob_strings.split('\n'):
        mat_obj = parse_regex_train.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                train_objf[int(groups[0])] = groups[2]
    if not train_objf:
        raise KaldiLogParseException("Could not find any lines with {k} in "
                " {l}".format(k=key, l=train_prob_files))

    for line in valid_prob_strings.split('\n'):
        mat_obj = parse_regex_valid.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                valid_objf[int(groups[0])] = groups[2]

    if not valid_objf:
        raise KaldiLogParseException("Could not find any lines with {k} in "
                " {l}".format(k=key, l=valid_prob_files))

    iters = list(set(valid_objf.keys()).intersection(train_objf.keys()))
    if not iters:
        raise KaldiLogParseException("Could not any common iterations with"
                " key {k} in both {tl} and {vl}".format(
                    k=key, tl=train_prob_files, vl=valid_prob_files))
    iters.sort()
    return map(lambda x: (int(x), float(train_objf[x]),
                          float(valid_objf[x])), iters)



def generate_acc_logprob_report(exp_dir, key="accuracy", output="output"):
    try:
        times = get_train_times(exp_dir)
    except:
        tb = traceback.format_exc()
        logger.warning("Error getting info from logs, exception was: " + tb)
        times = {}

    report = []
    report.append("%Iter\tduration\ttrain_objective\tvalid_objective\tdifference")
    try:
        if key == "rnnlm_objective":
            data = list(parse_rnnlm_prob_logs(exp_dir, 'objf'))
        else:
            data = list(parse_prob_logs(exp_dir, key, output))
    except:
        tb = traceback.format_exc()
        logger.warning("Error getting info from logs, exception was: " + tb)
        data = []
    for x in data:
        try:
            report.append("%d\t%s\t%g\t%g\t%g" % (x[0], str(times[x[0]]),
                                                  x[1], x[2], x[2]-x[1]))
        except KeyError, IndexError:
            continue

    total_time = 0
    for iter in times.keys():
        total_time += times[iter]
    report.append("Total training time is {0}\n".format(
                    str(datetime.timedelta(seconds=total_time))))
    return ["\n".join(report), times, data]
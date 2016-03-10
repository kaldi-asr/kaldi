# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.

from __future__ import division
import sys, glob, re, math, datetime, argparse
import imp

ntl = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

#exp/nnet3/lstm_self_repair_ld5_sp/log/progress.9.log:component name=Lstm3_i type=SigmoidComponent, dim=1280, self-repair-scale=1e-05, count=1.96e+05, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.05,0.09,0.11,0.15 0.19,0.27,0.50,0.72,0.83 0.88,0.92,0.94,0.99), mean=0.502, stddev=0.23], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.009,0.04,0.05,0.06 0.08,0.10,0.14,0.17,0.18 0.19,0.20,0.20,0.21), mean=0.134, stddev=0.0397]
def ParseProgressLogsForNonlinearityStats(exp_dir):
    progress_log_files = "%s/log/progress.*.log" % (exp_dir)
    stats_per_component_per_iter = {}

    progress_log_lines  = ntl.RunKaldiCommand('grep -e "value-avg.*deriv-avg" {0}'.format(progress_log_files))[0]

    parse_regex = re.compile(".*progress.([0-9]+).log:component name=(.+) type=(.*)Component,.*value-avg=\[.*mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\].*deriv-avg=\[.*mean=([0-9\.\-e]+), stddev=([0-9\.e\-]+)\]")
    for line in progress_log_lines.split("\n") :
        mat_obj = parse_regex.search(line)
        if mat_obj is None:
            continue
        groups = mat_obj.groups()
        # groups  = ('9', 'Lstm3_i', 'Sigmoid', '0.502', '0.23', '0.134', '0.0397')
        iteration = int(groups[0])
        component_name = groups[1]
        component_type = groups[2]
        value_mean = float(groups[3])
        value_stddev = float(groups[4])
        deriv_mean = float(groups[5])
        deriv_stddev = float(groups[6])
        try:
            stats_per_component_per_iter[component_name]['stats'][iteration] = [value_mean, value_stddev, deriv_mean, deriv_stddev]
        except KeyError:
            stats_per_component_per_iter[component_name] = {}
            stats_per_component_per_iter[component_name]['type'] = component_type
            stats_per_component_per_iter[component_name]['stats'] = {}
            stats_per_component_per_iter[component_name]['stats'][iteration] = [value_mean, value_stddev, deriv_mean, deriv_stddev]

    return stats_per_component_per_iter

def ParseDifferenceString(string):
    dict = {}
    for parts in string.split():
        sub_parts = parts.split(":")
        dict[sub_parts[0]] = float(sub_parts[1])
    return dict

#exp/chain/cwrnn_trial2_ld5_sp/log/progress.245.log:LOG (nnet3-show-progress:main():nnet3-show-progress.cc:144) Relative parameter differences per layer are [ Cwrnn1_T3_W_r:0.0171537 Cwrnn1_T3_W_x:1.33338e-07 Cwrnn1_T2_W_r:0.048075 Cwrnn1_T2_W_x:1.34088e-07 Cwrnn1_T1_W_r:0.0157277 Cwrnn1_T1_W_x:0.0212704 Final_affine:0.0321521 Cwrnn2_T3_W_r:0.0212082 Cwrnn2_T3_W_x:1.33691e-07 Cwrnn2_T2_W_r:0.0212978 Cwrnn2_T2_W_x:1.33401e-07 Cwrnn2_T1_W_r:0.014976 Cwrnn2_T1_W_x:0.0233588 Cwrnn3_T3_W_r:0.0237165 Cwrnn3_T3_W_x:1.33184e-07 Cwrnn3_T2_W_r:0.0239754 Cwrnn3_T2_W_x:1.3296e-07 Cwrnn3_T1_W_r:0.0194809 Cwrnn3_T1_W_x:0.0271934 ]
def ParseProgressLogsForParamDiff(exp_dir, pattern):
    if pattern not in set(["Relative parameter differences", "Parameter differences"]):
        raise Exception("Unknown value for pattern : {0}".format(pattern))

    progress_log_files = "%s/log/progress.*.log" % (exp_dir)
    progress_per_iter = {}
    component_names = set([])
    progress_log_lines = ntl.RunKaldiCommand('grep -e "{0}" {1}'.format(pattern, progress_log_files))[0]
    parse_regex = re.compile(".*progress\.([0-9]+)\.log:LOG.*{0}.*\[(.*)\]".format(pattern))
    for line in progress_log_lines.split("\n") :
        mat_obj = parse_regex.search(line)
        if mat_obj is None:
            continue
        groups = mat_obj.groups()
        iteration = groups[0]
        differences = ParseDifferenceString(groups[1])
        component_names  = component_names.union(differences.keys())
        progress_per_iter[int(iteration)] = differences

    component_names = list(component_names)
    component_names.sort()
    # rearranging the data into an array
    data = []
    data.append(["iteration"]+component_names)
    max_iter = max(progress_per_iter.keys())
    for iter in range(max_iter + 1):
        try:
            component_dict = progress_per_iter[iter]
        except KeyError:
            continue
        iter_values = []
        for component_name in component_names:
            try:
                iter_values.append(component_dict[component_name])
            except KeyError:
                # the component was not found this iteration, may be because of layerwise discriminative training
                iter_values.append(0)
        data.append([iter] + iter_values)

    return data

def ParseTrainLogs(exp_dir):
  train_log_files = "%s/log/train.*.log" % (exp_dir)
  train_log_lines = ntl.RunKaldiCommand('grep -e Accounting {0}'.format(train_log_files))[0]
  parse_regex = re.compile(".*train\.([0-9]+)\.([0-9]+)\.log:# Accounting: time=([0-9]+) thread.*")

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

def ParseProbLogs(exp_dir, key = 'accuracy'):
    train_prob_files = "%s/log/compute_prob_train.*.log" % (exp_dir)
    valid_prob_files = "%s/log/compute_prob_valid.*.log" % (exp_dir)
    train_prob_strings = ntl.RunKaldiCommand('grep -e {0} {1}'.format(key, train_prob_files), wait = True)[0]
    valid_prob_strings = ntl.RunKaldiCommand('grep -e {0} {1}'.format(key, valid_prob_files))[0]

    #LOG (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:149) Overall log-probability for 'output' is -0.399395 + -0.013437 = -0.412832 per frame, over 20000 fra
    #LOG (nnet3-chain-compute-prob:PrintTotalStats():nnet-chain-diagnostics.cc:144) Overall log-probability for 'output' is -0.307255 per frame, over 20000 frames.
    parse_regex = re.compile(".*compute_prob_.*\.([0-9]+).log:LOG .nnet3.*compute-prob:PrintTotalStats..:nnet.*diagnostics.cc:[0-9]+. Overall ([a-zA-Z\-]+) for 'output'.*is ([0-9.\-e]+) .*per frame")
    train_loss={}
    valid_loss={}


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
    return map(lambda x: (int(x), float(train_loss[x]), float(valid_loss[x])), iters)

def GenerateAccuracyReport(exp_dir, key = "accuracy"):
    times = ParseTrainLogs(exp_dir)
    data = ParseProbLogs(exp_dir, key)
    report = []
    report.append("%Iter\tduration\ttrain_loss\tvalid_loss\tdifference")
    for x in data:
        try:
            report.append("%d\t%s\t%g\t%g\t%g" % (x[0], str(times[x[0]]), x[1], x[2], x[2]-x[1]))
        except KeyError:
            continue

    total_time = 0
    for iter in times.keys():
        total_time += times[iter]
    report.append("Total training time is {0}\n".format(str(datetime.timedelta(seconds = total_time))))
    return ["\n".join(report), times, data]

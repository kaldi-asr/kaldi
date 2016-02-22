#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.

import warnings
import imp
import argparse
import os
import errno
import logging

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plot = True
except ImportError:
    warnings.warn("""
This script requires matplotlib and numpy. Please install them to generate plots. Proceeding with generation of tables.
If you are on a cluster where you do not have admin rights you could try using virtualenv.""")

nlp = imp.load_source('nlp', 'steps/nnet3/report/nnet3_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Generating plots')




def GetArgs():
    parser = argparse.ArgumentParser(description="Parses the training logs and generates a variety of plots.")
    parser.add_argument("exp_dir", help="experiment directory, e.g. exp/nnet3/tdnn")
    parser.add_argument("output_dir", help="experiment directory, e.g. exp/nnet3/tdnn/report")

    args = parser.parse_args()

    return args

def GeneratePlots(exp_dir, output_dir):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise e

    logger.info("Generating accuracy plots")
    [accuracy_report, accuracy_times, accuracy_data] = nlp.GenerateAccuracyReport(exp_dir, "accuracy")
    acc_file = open("{0}/accuracy.log".format(output_dir), "w")
    acc_file.write(accuracy_report)
    acc_file.close()
    if plot:
        data = np.array(accuracy_data)
        plt.clf()
        plt.plot(data[:, 0], data[:, 1], 'r', data[:, 0], data[:, 2], 'b')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig('{0}/accuracy.pdf'.format(output_dir))

    logger.info("Generating log-likelihood plots")
    [loglike_report, loglike_times, loglike_data] = nlp.GenerateAccuracyReport(exp_dir, "log-likelihood")
    ll_file = open("{0}/loglikelihood.log".format(output_dir), "w")
    ll_file.write(loglike_report)
    if plot:
        data = np.array(loglike_data)
        plt.clf()
        plt.plot(data[:, 0], data[:, 1], 'r', data[:, 0], data[:, 2], 'b')
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')
        plt.grid(True)
        plt.savefig('{0}/loglikelihood.pdf'.format(output_dir))

    logger.info("Generating non-linearity stats plots")
    stats_per_component_per_iter = nlp.ParseProgressLogsForNonlinearityStats(exp_dir)
    component_names = stats_per_component_per_iter.keys()
    for component_name in component_names:
        comp_data = stats_per_component_per_iter[component_name]
        comp_type = comp_data['type']
        comp_stats = comp_data['stats']
        #stats are stored as [value_mean, value_stddev, deriv_mean, deriv_stddev]
        file = open("{dir}/nonlinstats_{comp_name}.log".format(dir = output_dir, comp_name = component_name), "w")
        file.write("Iteration\tValueMean\tValueStddev\tDerivMean\tDerivStddev\n")
        iters = comp_stats.keys()
        iters.sort()
        iter_stats = []
        iter_stat_report = ""
        for iter in iters:
            iter_stats.append([iter] + comp_stats[iter])
            iter_stat_report += "\t".join(map(lambda x: str(x), iter_stats[-1])) + "\n"
        file.close()

        if plot:
            data = np.array(iter_stats)
            plt.clf()
            ax = plt.subplot(211)
            mp, = ax.plot(data[:,0], data[:,1], color='blue', label="Mean")
            msph, = ax.plot(data[:,0], data[:,1] + data[:,2], color='red', label = "Mean+Stddev")
            mspl, = ax.plot(data[:,0], data[:,1] - data[:,2], color='red', label = "Mean-Stddev")
            lgd = plt.legend(handles=[mp, msph, mspl])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
                               ncol=3, mode="expand", borderaxespad=0.)

            ax.set_ylabel('Mean-{0}'.format(comp_type))
            ax.grid(True)

            ax = plt.subplot(212)
            mp, = ax.plot(data[:,0], data[:,3], color='blue', label="Mean")
            msph, = ax.plot(data[:,0], data[:,3] + data[:,4], color='red', label = "Mean+Stddev")
            mspl, = ax.plot(data[:,0], data[:,3] - data[:,4], color='red', label = "Mean-Stddev")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Derivative-{0}'.format(comp_type))
            ax.grid(True)
            plt.savefig('{dir}/nonlinstats_{comp_name}.pdf'.format(dir = output_dir, comp_name = component_name), bbox_extra_artists=(lgd,), bbox_inches='tight')


    logger.info("Generating parameter difference files")
    # Parameter changes
    key_file = {"Parameter differences":"parameter.diff",
                "Relative parameter differences":"relative_parameter.diff"}
    for key in key_file.keys():
        file = open("{0}/{1}".format(output_dir, key_file[key]), "w")
        data = nlp.ParseProgressLogsForParamDiff(exp_dir, key)
        for row in data:
            file.write(" ".join(map(lambda x:str(x),row))+"\n")
        file.close()

def Main():
    args = GetArgs()
    GeneratePlots(args.exp_dir, args.output_dir)

if __name__ == "__main__":
    Main()

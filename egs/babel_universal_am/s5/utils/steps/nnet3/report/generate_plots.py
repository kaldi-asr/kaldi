#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti
#           2016    Vimal Manohar
# Apache 2.0.

import argparse
import errno
import logging
import os
import re
import sys
import warnings

sys.path.insert(0, 'steps')
import libs.nnet3.report.log_parse as log_parse
import libs.common as common_lib

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    g_plot = True
except ImportError:
    warnings.warn(
        """This script requires matplotlib and numpy.
        Please install them to generate plots.
        Proceeding with generation of tables.
        If you are on a cluster where you do not have admin rights you could
        try using virtualenv.""")
    g_plot = False


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Generating plots')


def get_args():
    parser = argparse.ArgumentParser(
        description="""Parses the training logs and generates a variety of
        plots.
        e.g. (deprecated): steps/nnet3/report/generate_plots.py
        --comparison-dir exp/nnet3/tdnn1 --comparison-dir exp/nnet3/tdnn2
        exp/nnet3/tdnn exp/nnet3/tdnn/report
        or (current): steps/nnet3/report/generate_plots.py
        exp/nnet3/tdnn exp/nnet3/tdnn1 exp/nnet3/tdnn2 exp/nnet3/tdnn/report.
        Look for the report.pdf in the output (report) directory.""")

    parser.add_argument("--comparison-dir", type=str, action='append',
                        help="other experiment directories for comparison. "
                        "These will only be used for plots, not tables"
                        "Note: this option is deprecated.")
    parser.add_argument("--start-iter", type=int,
                        help="Iteration from which plotting will start",
                        default=1)
    parser.add_argument("--is-chain", type=str, default=False,
                        action=common_lib.StrToBoolAction,
                        help="True if directory contains chain models")
    parser.add_argument("--output-nodes", type=str, default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="""List of space separated
                        <output-node>:<objective-type> entities,
                        one for each output node""")
    parser.add_argument("exp_dir", nargs='+',
                        help="the first dir is the experiment directory, "
                        "e.g. exp/nnet3/tdnn, the rest dirs (if exist) "
                        "are other experiment directories for comparison.")
    parser.add_argument("output_dir",
                        help="experiment directory, "
                        "e.g. exp/nnet3/tdnn/report")

    args = parser.parse_args()
    if (args.comparison_dir is not None and len(args.comparison_dir) > 6) or \
    (args.exp_dir is not None and len(args.exp_dir) > 7):
        raise Exception(
            """max 6 comparison directories can be specified.
            If you want to compare with more comparison_dir, you would have to
            carefully tune the plot_colors variable which specified colors used
            for plotting.""")
    assert args.start_iter >= 1
    return args


g_plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow', 'cyan']

class LatexReport:
    """Class for writing a Latex report"""

    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.document = []
        self.document.append("""
\documentclass[prl,10pt,twocolumn]{revtex4}
\usepackage{graphicx}    % Used to import the graphics
\\begin{document}
""")

    def add_figure(self, figure_pdf, title):
        """we will have keep extending this replacement list based on errors
        during compilation escaping underscores in the title"""
        title = "\\texttt{"+re.sub("_", "\_", title)+"}"
        fig_latex = """
%...
\\newpage
\\begin{figure}[h]
  \\begin{center}
    \caption{""" + title + """}
    \includegraphics[width=\\textwidth]{""" + figure_pdf + """}
  \end{center}
\end{figure}
\clearpage
%...
"""
        self.document.append(fig_latex)

    def close(self):
        self.document.append("\end{document}")
        return self.compile()

    def compile(self):
        root, ext = os.path.splitext(self.pdf_file)
        dir_name = os.path.dirname(self.pdf_file)
        latex_file = root + ".tex"
        lat_file = open(latex_file, "w")
        lat_file.write("\n".join(self.document))
        lat_file.close()
        logger.info("Compiling the latex report.")
        try:
            common_lib.execute_command(
                "pdflatex -interaction=batchmode "
                "-output-directory={0} {1}".format(dir_name, latex_file))
        except Exception as e:
            logger.warning("There was an error compiling the latex file {0}, "
                           "please do it manually: {1}".format(latex_file, e))
            return False
        return True


def latex_compliant_name(name_string):
    """this function is required as latex does not allow all the component names
    allowed by nnet3.
    Identified incompatibilities :
        1. latex does not allow dot(.) in file names
    """
    node_name_string = re.sub("\.", "_dot_", name_string)

    return node_name_string


def generate_acc_logprob_plots(exp_dir, output_dir, plot, key='accuracy',
        file_basename='accuracy', comparison_dir=None,
        start_iter=1, latex_report=None, output_name='output'):

    assert start_iter >= 1

    if plot:
        fig = plt.figure()
        plots = []

    comparison_dir = [] if comparison_dir is None else comparison_dir
    dirs = [exp_dir] + comparison_dir
    index = 0
    for dir in dirs:
        [report, times, data] = log_parse.generate_acc_logprob_report(dir, key,
                output_name)
        if index == 0:
            # this is the main experiment directory
            with open("{0}/{1}.log".format(output_dir,
                                           file_basename), "w") as f:
                f.write(report)

        if plot:
            color_val = g_plot_colors[index]
            data = np.array(data)
            if data.shape[0] == 0:
                logger.warning("Couldn't find any rows for the"
                               "accuracy/log-probability plot, not generating it")
                return
            data = data[data[:, 0] >= start_iter, :]
            plot_handle, = plt.plot(data[:, 0], data[:, 1], color=color_val,
                                    linestyle="--",
                                    label="train {0}".format(dir))
            plots.append(plot_handle)
            plot_handle, = plt.plot(data[:, 0], data[:, 2], color=color_val,
                                    label="valid {0}".format(dir))
            plots.append(plot_handle)
        index += 1
    if plot:
        plt.xlabel('Iteration')
        plt.ylabel(key)
        lgd = plt.legend(handles=plots, loc='lower center',
                         bbox_to_anchor=(0.5, -0.2 + len(dirs) * -0.1),
                         ncol=1, borderaxespad=0.)
        plt.grid(True)
        fig.suptitle("{0} plot for {1}".format(key, output_name))
        figfile_name = '{0}/{1}_{2}.pdf'.format(
            output_dir, file_basename,
            latex_compliant_name(output_name))
        plt.savefig(figfile_name, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        if latex_report is not None:
            latex_report.add_figure(
                figfile_name,
                "Plot of {0} vs iterations for {1}".format(key, output_name))


# The name of five gates of lstmp
g_lstm_gate = ['i_t_sigmoid', 'f_t_sigmoid', 'c_t_tanh', 'o_t_sigmoid', 'm_t_tanh']

# The "extra" item looks like a placeholder. As each unit in python plot is
# composed by a legend_handle(linestyle) and a legend_label(description).
# For the unit which doesn't have linestyle, we use the "extra" placeholder.
extra = Rectangle((0, 0), 1, 1, facecolor="w", fill=False, edgecolor='none', linewidth=0)

# This function is used to insert a column to the legend, the column_index is 1-based
def insert_a_column_legend(legend_handle, legend_label, lp, mp, hp,
        dir, prefix_length, column_index):
    handle = [extra, lp, mp, hp]
    label = ["[1]{0}".format(dir[prefix_length:]), "", "", ""]
    for row in range(1,5):
        legend_handle.insert(column_index*row-1, handle[row-1])
        legend_label.insert(column_index*row-1, label[row-1])


# This function is used to plot a normal nonlinearity component or a gate of lstmp
def plot_a_nonlin_component(fig, dirs, stat_tables_per_component_per_dir,
        component_name, common_prefix, prefix_length, component_type,
        start_iter, gate_index=0):
    fig.clf()
    index = 0
    legend_handle = [extra, extra, extra, extra]
    legend_label = ["", '5th percentile', '50th percentile', '95th percentile']

    for dir in dirs:
        color_val = g_plot_colors[index]
        index += 1
        try:
            iter_stats = (stat_tables_per_component_per_dir[dir][component_name])
        except KeyError:
            # this component is not available in this network so lets
            # not just plot it
            insert_a_column_legend(legend_handle, legend_label, lp, mp, hp,
                    dir, prefix_length, index+1)
            continue

        data = np.array(iter_stats)
        data = data[data[:, 0] >= start_iter, :]
        ax = plt.subplot(211)
        lp, = ax.plot(data[:, 0], data[:, gate_index*10+5], color=color_val,
                linestyle='--')
        mp, = ax.plot(data[:, 0], data[:, gate_index*10+6], color=color_val,
                linestyle='-')
        hp, = ax.plot(data[:, 0], data[:, gate_index*10+7], color=color_val,
                linestyle='--')
        insert_a_column_legend(legend_handle, legend_label, lp, mp, hp,
                dir, prefix_length, index+1)

        ax.set_ylabel('Value-{0}'.format(component_type))
        ax.grid(True)

        ax = plt.subplot(212)
        lp, = ax.plot(data[:, 0], data[:, gate_index*10+8], color=color_val,
                linestyle='--')
        mp, = ax.plot(data[:, 0], data[:, gate_index*10+9], color=color_val,
                linestyle='-')
        hp, = ax.plot(data[:, 0], data[:, gate_index*10+10], color=color_val,
                linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Derivative-{0}'.format(component_type))
        ax.grid(True)

    lgd = plt.legend(legend_handle, legend_label, loc='lower center',
            bbox_to_anchor=(0.5 , -0.5 + len(dirs) * -0.2),
            ncol=4, handletextpad = -2, title="[1]:{0}".format(common_prefix),
            borderaxespad=0.)
    plt.grid(True)
    return lgd


# This function is used to generate the statistic plots of nonlinearity component
# Mainly divided into the following steps:
# 1) With log_parse function, we get the statistics from each directory.
# 2) Convert the collected nonlinearity statistics into the tables. Each table
#    contains all the statistics in each component of each directory.
# 3) The statistics of each component are stored into corresponding log files.
#    Each line of the log file contains the statistics of one iteration.
# 4) Plot the "Per-dimension average-(value, derivative) percentiles" figure
#    for each nonlinearity component.
def generate_nonlin_stats_plots(exp_dir, output_dir, plot, comparison_dir=None,
                                start_iter=1, latex_report=None):
    assert start_iter >= 1

    comparison_dir = [] if comparison_dir is None else comparison_dir
    dirs = [exp_dir] + comparison_dir
    index = 0
    stats_per_dir = {}

    for dir in dirs:
        stats_per_component_per_iter = (
            log_parse.parse_progress_logs_for_nonlinearity_stats(dir))
        for key in stats_per_component_per_iter:
            if len(stats_per_component_per_iter[key]['stats']) == 0:
                logger.warning("Couldn't find any rows for the"
                               "nonlin stats plot, not generating it")
        stats_per_dir[dir] = stats_per_component_per_iter
    # convert the nonlin stats into tables
    stat_tables_per_component_per_dir = {}
    for dir in dirs:
        stats_per_component_per_iter = stats_per_dir[dir]
        component_names = stats_per_component_per_iter.keys()
        stat_tables_per_component = {}
        for component_name in component_names:
            comp_data = stats_per_component_per_iter[component_name]
            comp_type = comp_data['type']
            comp_stats = comp_data['stats']
            iters = comp_stats.keys()
            iters.sort()
            iter_stats = []
            for iter in iters:
                iter_stats.append([iter] + comp_stats[iter])
            stat_tables_per_component[component_name] = iter_stats
        stat_tables_per_component_per_dir[dir] = stat_tables_per_component

    main_stat_tables = stat_tables_per_component_per_dir[exp_dir]
    for component_name in main_stat_tables.keys():
        # this is the main experiment directory
        with open("{dir}/nonlinstats_{comp_name}.log".format(
                    dir=output_dir, comp_name=component_name), "w") as f:
            f.write("Iteration\tValueMean\tValueStddev\tDerivMean\tDerivStddev\t"
                               "Value_5th\tValue_50th\tValue_95th\t"
                               "Deriv_5th\tDeriv_50th\tDeriv_95th\n")
            iter_stat_report = []
            iter_stats = main_stat_tables[component_name]
            for row in iter_stats:
                iter_stat_report.append("\t".join([str(x) for x in row]))
            f.write("\n".join(iter_stat_report))
            f.close()
    if plot:
        main_component_names = main_stat_tables.keys()
        main_component_names.sort()

        plot_component_names = set(main_component_names)
        for dir in dirs:
            component_names = set(stats_per_dir[dir].keys())
            plot_component_names = plot_component_names.intersection(
                component_names)
        plot_component_names = list(plot_component_names)
        plot_component_names.sort()
        if plot_component_names != main_component_names:
            logger.warning("""The components in all the neural networks in the
            given experiment dirs are not the same, so comparison plots are
            provided only for common component names. Make sure that these are
            comparable experiments before analyzing these plots.""")

        fig = plt.figure()

        common_prefix = os.path.commonprefix(dirs)
        prefix_length = common_prefix.rfind('/')
        common_prefix = common_prefix[0:prefix_length]

        for component_name in main_component_names:
            if stats_per_dir[exp_dir][component_name]['type'] == 'LstmNonlinearity':
                for i in range(0,5):
                    component_type = 'Lstm-' + g_lstm_gate[i]
                    lgd = plot_a_nonlin_component(fig, dirs,
                            stat_tables_per_component_per_dir, component_name,
                            common_prefix, prefix_length, component_type, start_iter, i)
                    fig.suptitle("Per-dimension average-(value, derivative) percentiles for "
                         "{component_name}-{gate}".format(component_name=component_name, gate=g_lstm_gate[i]))
                    comp_name = latex_compliant_name(component_name)
                    figfile_name = '{dir}/nonlinstats_{comp_name}_{gate}.pdf'.format(
                        dir=output_dir, comp_name=comp_name, gate=g_lstm_gate[i])
                    fig.savefig(figfile_name, bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
                    if latex_report is not None:
                        latex_report.add_figure(
                        figfile_name,
                        "Per-dimension average-(value, derivative) percentiles for "
                        "{0}-{1}".format(component_name, g_lstm_gate[i]))
            else:
                component_type = stats_per_dir[exp_dir][component_name]['type']
                lgd = plot_a_nonlin_component(fig, dirs,
                        stat_tables_per_component_per_dir,component_name,
                        common_prefix, prefix_length, component_type, start_iter, 0)
                fig.suptitle("Per-dimension average-(value, derivative) percentiles for "
                         "{component_name}".format(component_name=component_name))
                comp_name = latex_compliant_name(component_name)
                figfile_name = '{dir}/nonlinstats_{comp_name}.pdf'.format(
                    dir=output_dir, comp_name=comp_name)
                fig.savefig(figfile_name, bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
                if latex_report is not None:
                    latex_report.add_figure(
                    figfile_name,
                    "Per-dimension average-(value, derivative) percentiles for "
                    "{0}".format(component_name))



def generate_clipped_proportion_plots(exp_dir, output_dir, plot,
                                      comparison_dir=None, start_iter=1,
                                      latex_report=None):
    assert(start_iter >= 1)

    comparison_dir = [] if comparison_dir is None else comparison_dir
    dirs = [exp_dir] + comparison_dir
    index = 0
    stats_per_dir = {}
    for dir in dirs:
        try:
            stats_per_dir[dir] = (
                log_parse.parse_progress_logs_for_clipped_proportion(dir))
        except log_parse.MalformedClippedProportionLineException as e:
            raise e
        except common_lib.KaldiCommandException as e:
            warnings.warn("Could not extract the clipped proportions for {0},"
                          " this might be because there are no "
                          "ClipGradientComponents.".format(dir))
            continue
        if len(stats_per_dir[dir]) == 0:
            logger.warning("Couldn't find any rows for the"
                           "clipped proportion plot, not generating it")
    try:
        main_cp_stats = stats_per_dir[exp_dir]['table']
    except KeyError:
        warnings.warn("The main experiment directory {0} does not have "
                      "clipped proportions. So not generating clipped "
                      "proportion plots.".format(exp_dir))
        return

    # this is the main experiment directory
    file = open("{dir}/clipped_proportion.log".format(dir=output_dir), "w")
    iter_stat_report = ""
    for row in main_cp_stats:
        iter_stat_report += "\t".join(map(lambda x: str(x), row)) + "\n"
    file.write(iter_stat_report)
    file.close()

    if plot:
        main_component_names = (
            stats_per_dir[exp_dir]['cp_per_iter_per_component'].keys())
        main_component_names.sort()
        plot_component_names = set(main_component_names)
        for dir in dirs:
            try:
                component_names = set(
                    stats_per_dir[dir]['cp_per_iter_per_component'].keys())
                plot_component_names = (
                    plot_component_names.intersection(component_names))
            except KeyError:
                continue
        plot_component_names = list(plot_component_names)
        plot_component_names.sort()
        if plot_component_names != main_component_names:
            logger.warning(
                """The components in all the neural networks in the given
                experiment dirs are not the same, so comparison plots are
                provided only for common component names. Make sure that these
                are comparable experiments before analyzing these plots.""")

        fig = plt.figure()
        for component_name in main_component_names:
            fig.clf()
            index = 0
            plots = []
            for dir in dirs:
                color_val = g_plot_colors[index]
                index += 1
                try:
                    iter_stats = stats_per_dir[dir][
                        'cp_per_iter_per_component'][component_name]
                except KeyError:
                    # this component is not available in this network so lets
                    # not just plot it
                    continue

                data = np.array(iter_stats)
                data = data[data[:, 0] >= start_iter, :]
                ax = plt.subplot(111)
                mp, = ax.plot(data[:, 0], data[:, 1], color=color_val,
                              label="Clipped Proportion {0}".format(dir))
                plots.append(mp)
                ax.set_ylabel('Clipped Proportion')
                ax.set_ylim([0, 1.2])
                ax.grid(True)
            lgd = plt.legend(handles=plots, loc='lower center',
                             bbox_to_anchor=(0.5, -0.5 + len(dirs) * -0.2),
                             ncol=1, borderaxespad=0.)
            plt.grid(True)
            fig.suptitle("Clipped-proportion value at {comp_name}".format(
                            comp_name=component_name))
            comp_name = latex_compliant_name(component_name)
            figfile_name = '{dir}/clipped_proportion_{comp_name}.pdf'.format(
                dir=output_dir, comp_name=comp_name)
            fig.savefig(figfile_name, bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
            if latex_report is not None:
                latex_report.add_figure(
                    figfile_name,
                    "Clipped proportion at {0}".format(component_name))


def generate_parameter_diff_plots(exp_dir, output_dir, plot,
                                  comparison_dir=None, start_iter=1,
                                  latex_report=None):
    # Parameter changes
    assert start_iter >= 1

    comparison_dir = [] if comparison_dir is None else comparison_dir
    dirs = [exp_dir] + comparison_dir
    index = 0
    stats_per_dir = {}
    key_file = {"Parameter differences": "parameter.diff",
                "Relative parameter differences": "relative_parameter.diff"}
    stats_per_dir = {}
    for dir in dirs:
        stats_per_dir[dir] = {}
        for key in key_file:
            stats_per_dir[dir][key] = (
                log_parse.parse_progress_logs_for_param_diff(dir, key))

    # write down the stats for the main experiment directory
    for diff_type in key_file:
        with open("{0}/{1}".format(output_dir, key_file[diff_type]), "w") as f:
            diff_per_component_per_iter = (
                stats_per_dir[exp_dir][diff_type]['progress_per_component'])
            component_names = (
                stats_per_dir[exp_dir][diff_type]['component_names'])
            max_iter = stats_per_dir[exp_dir][diff_type]['max_iter']
            f.write(" ".join(["Iteration"] + component_names)+"\n")
            total_missing_iterations = 0
            gave_user_warning = False
            for iter in range(max_iter + 1):
                iter_data = [str(iter)]
                for c in component_names:
                    try:
                        iter_data.append(
                            str(diff_per_component_per_iter[c][iter]))
                    except KeyError:
                        total_missing_iterations += 1
                        iter_data.append("NA")
                if (total_missing_iterations/len(component_names) > 20
                        and not gave_user_warning):
                    logger.warning("There are more than {0} missing "
                                   "iterations per component. "
                                   "Something might be wrong.".format(
                                       total_missing_iterations
                                       / len(component_names)))
                    gave_user_warning = True

                f.write(" ".join(iter_data)+"\n")

    if plot:
        # get the component names
        diff_type = key_file.keys()[0]
        main_component_names = stats_per_dir[exp_dir][diff_type][
            'progress_per_component'].keys()
        main_component_names.sort()
        plot_component_names = set(main_component_names)

        for dir in dirs:
            try:
                component_names = set(stats_per_dir[dir][diff_type][
                    'progress_per_component'].keys())
                plot_component_names = plot_component_names.intersection(
                    component_names)
            except KeyError:
                continue
        plot_component_names = list(plot_component_names)
        plot_component_names.sort()
        if plot_component_names != main_component_names:
            logger.warning("The components in all the neural networks in the "
                           "given experiment dirs are not the same, "
                           "so comparison plots are provided only for common "
                           "component names. "
                           "Make sure that these are comparable experiments "
                           "before analyzing these plots.")

        assert main_component_names

        fig = plt.figure()
        logger.info("Generating parameter-difference plots for the "
                    "following components:{0}".format(
                        ', '.join(main_component_names)))

        for component_name in main_component_names:
            fig.clf()
            index = 0
            plots = []
            for dir in dirs:
                color_val = g_plot_colors[index]
                index += 1
                iter_stats = []
                try:
                    for diff_type in ['Parameter differences',
                                      'Relative parameter differences']:
                        iter_stats.append(np.array(
                            sorted(stats_per_dir[dir][diff_type][
                                'progress_per_component'][
                                    component_name].items())))
                except KeyError as e:
                    # this component is not available in this network so lets
                    # not just plot it
                    if dir == exp_dir:
                        raise Exception("No parameter differences were "
                                        "available even in the main "
                                        "experiment dir for the component "
                                        "{0}. Something went wrong: "
                                        "{1}.".format(
                                            component_name, str(e)))
                    continue
                ax = plt.subplot(211)
                mp, = ax.plot(iter_stats[0][:, 0], iter_stats[0][:, 1],
                              color=color_val,
                              label="Parameter Differences {0}".format(dir))
                plots.append(mp)
                ax.set_ylabel('Parameter Differences')
                ax.grid(True)

                ax = plt.subplot(212)
                mp, = ax.plot(iter_stats[1][:, 0], iter_stats[1][:, 1],
                              color=color_val,
                              label="Relative Parameter "
                                    "Differences {0}".format(dir))
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Relative Parameter Differences')
                ax.grid(True)

            lgd = plt.legend(handles=plots, loc='lower center',
                             bbox_to_anchor=(0.5, -0.5 + len(dirs) * -0.2),
                             ncol=1, borderaxespad=0.)
            plt.grid(True)
            fig.suptitle("Parameter differences at {comp_name}".format(
                comp_name=component_name))
            comp_name = latex_compliant_name(component_name)
            figfile_name = '{dir}/param_diff_{comp_name}.pdf'.format(
                dir=output_dir, comp_name=comp_name)
            fig.savefig(figfile_name, bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
            if latex_report is not None:
                latex_report.add_figure(
                    figfile_name,
                    "Parameter differences at {0}".format(component_name))


def generate_plots(exp_dir, output_dir, output_names, comparison_dir=None,
                   start_iter=1):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise e
    if g_plot:
        latex_report = LatexReport("{0}/report.pdf".format(output_dir))
    else:
        latex_report = None

    for (output_name, objective_type) in output_names:
        if objective_type == "linear":
            logger.info("Generating accuracy plots")
            generate_acc_logprob_plots(
                exp_dir, output_dir, g_plot, key='accuracy',
                file_basename='accuracy', comparison_dir=comparison_dir,
                start_iter=start_iter,
                latex_report=latex_report, output_name=output_name)

            logger.info("Generating log-likelihood plots")
            generate_acc_logprob_plots(
                exp_dir, output_dir, g_plot, key='log-likelihood',
                file_basename='loglikelihood', comparison_dir=comparison_dir,
                start_iter=start_iter,
                latex_report=latex_report, output_name=output_name)
        elif objective_type == "chain":
            logger.info("Generating log-probability plots")
            generate_acc_logprob_plots(
                exp_dir, output_dir, g_plot,
                key='log-probability', file_basename='log_probability',
                comparison_dir=comparison_dir, start_iter=start_iter,
                latex_report=latex_report, output_name=output_name)
        else:
            logger.info("Generating " + objective_type + " objective plots")
            generate_acc_logprob_plots(
                exp_dir, output_dir, g_plot, key='objective',
                file_basename='objective', comparison_dir=comparison_dir,
                start_iter=start_iter,
                latex_report=latex_report, output_name=output_name)

    logger.info("Generating non-linearity stats plots")
    generate_nonlin_stats_plots(
        exp_dir, output_dir, g_plot, comparison_dir=comparison_dir,
        start_iter=start_iter, latex_report=latex_report)

    logger.info("Generating clipped-proportion plots")
    generate_clipped_proportion_plots(
        exp_dir, output_dir, g_plot, comparison_dir=comparison_dir,
        start_iter=start_iter, latex_report=latex_report)

    logger.info("Generating parameter difference plots")
    generate_parameter_diff_plots(
        exp_dir, output_dir, g_plot, comparison_dir=comparison_dir,
        start_iter=start_iter, latex_report=latex_report)

    if g_plot and latex_report is not None:
        has_compiled = latex_report.close()
        if has_compiled:
            logger.info("Report has been generated. "
                        "You can find it at the location "
                        "{0}".format("{0}/report.pdf".format(output_dir)))


def main():
    args = get_args()

    output_nodes = []

    if args.output_nodes is not None:
        nodes = args.output_nodes.split(' ')
        for n in nodes:
            parts = n.split(':')
            assert len(parts) == 2
            output_nodes.append(tuple(parts))
    elif args.is_chain:
        output_nodes.append(('output', 'chain'))
    else:
        output_nodes.append(('output', 'linear'))

    if args.comparison_dir is not None:
      generate_plots(args.exp_dir[0], args.output_dir, output_nodes,
                     comparison_dir=args.comparison_dir,
                     start_iter=args.start_iter)
    else:
      if len(args.exp_dir) == 1:
        generate_plots(args.exp_dir[0], args.output_dir, output_nodes,
                       start_iter=args.start_iter)
      if len(args.exp_dir) > 1:
        generate_plots(args.exp_dir[0], args.output_dir, output_nodes,
                       comparison_dir=args.exp_dir[1:],
                       start_iter=args.start_iter)


if __name__ == "__main__":
    main()

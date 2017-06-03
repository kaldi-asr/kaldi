

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
#           2017 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

""" This module contains several utility functions and classes that are
commonly used in many kaldi python scripts.
"""

import argparse
import logging
import math
import os
import subprocess
import threading

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def send_mail(message, subject, email_id):
    try:
        subprocess.Popen(
            'echo "{message}" | mail -s "{subject}" {email}'.format(
                message=message,
                subject=subject,
                email=email_id), shell=True)
    except Exception as e:
        logger.info("Unable to send mail due to error:\n {error}".format(
                        error=str(e)))
        pass


def str_to_bool(value):
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError


class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_bool(values))
        except ValueError:
            raise Exception(
                "Unknown value {0} for --{1}".format(values, self.dest))


class NullstrToNoneAction(argparse.Action):
    """ A custom action to convert empty strings passed by shell to None in
    python. This is necessary as shell scripts print null strings when a
    variable is not specified. We could use the more apt None in python. """

    def __call__(self, parser, namespace, values, option_string=None):
        if values.strip() == "":
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, values)


def check_if_cuda_compiled():
    p = subprocess.Popen("cuda-compiled")
    p.communicate()
    if p.returncode == 1:
        return False
    else:
        return True


def execute_command(command):
    """ Runs a kaldi job in the foreground and waits for it to complete; raises an
        exception if its return status is nonzero.  The command is executed in
        'shell' mode so 'command' can involve things like pipes.  Often,
        'command' will start with 'run.pl' or 'queue.pl'.  The stdout and stderr
        are merged with the calling process's stdout and stderr so they will
        appear on the screen.

        See also: get_command_stdout, background_command
    """
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode is not 0:
        raise Exception("Command exited with status {0}: {1}".format(
                p.returncode, command))


def get_command_stdout(command, require_zero_status = True):
    """ Executes a command and returns its stdout output as a string.  The
        command is executed with shell=True, so it may contain pipes and
        other shell constructs.

        If require_zero_stats is True, this function will raise an exception if
        the command has nonzero exit status.  If False, it just prints a warning
        if the exit status is nonzero.

        See also: execute_command, background_command
    """
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode is not 0:
        output = "Command exited with status {0}: {1}".format(
            p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logger.warning(output)
    return stdout if type(stdout) is str else stdout.decode()




def wait_for_background_commands():
    """ This waits for all threads to exit.  You will often want to
        run this at the end of programs that have launched background
        threads, so that the program will wait for its child processes
        to terminate before it dies."""
    for t in threading.enumerate():
        if not t == threading.current_thread():
            t.join()

def background_command(command, require_zero_status = False):
    """Executes a command in a separate thread, like running with '&' in the shell.
       If you want the program to die if the command eventually returns with
       nonzero status, then set require_zero_status to True.  'command' will be
       executed in 'shell' mode, so it's OK for it to contain pipes and other
       shell constructs.

       This function returns the Thread object created, just in case you want
       to wait for that specific command to finish.  For example, you could do:
             thread = background_command('foo | bar')
             # do something else while waiting for it to finish
             thread.join()

       See also:
         - wait_for_background_commands(), which can be used
           at the end of the program to wait for all these commands to terminate.
         - execute_command() and get_command_stdout(), which allow you to
           execute commands in the foreground.

    """

    p = subprocess.Popen(command, shell=True)
    thread = threading.Thread(target=background_command_waiter,
                              args=(command, p, require_zero_status))
    thread.daemon=True  # make sure it exits if main thread is terminated
                        # abnormally.
    thread.start()
    return thread


def background_command_waiter(command, popen_object, require_zero_status):
    """ This is the function that is called from background_command, in
        a separate thread."""

    popen_object.communicate()
    if popen_object.returncode is not 0:
        str = "Command exited with status {0}: {1}".format(
            popen_object.returncode, command)
        if require_zero_status:
            logger.error(str)
            # thread.interrupt_main() sends a KeyboardInterrupt to the main
            # thread, which will generally terminate the program.
            import thread
            thread.interrupt_main()
        else:
            logger.warning(str)


def get_number_of_leaves_from_tree(alidir):
    stdout = get_command_stdout(
        "tree-info {0}/tree 2>/dev/null | grep num-pdfs".format(alidir))
    parts = stdout.split()
    assert(parts[0] == "num-pdfs")
    num_leaves = int(parts[1])
    if num_leaves == 0:
        raise Exception("Number of leaves is 0")
    return num_leaves


def get_number_of_leaves_from_model(dir):
    stdout = get_command_stdout(
        "am-info {0}/final.mdl 2>/dev/null | grep -w pdfs".format(dir))
    parts = stdout.split()
    # number of pdfs 7115
    assert(' '.join(parts[0:3]) == "number of pdfs")
    num_leaves = int(parts[3])
    if num_leaves == 0:
        raise Exception("Number of leaves is 0")
    return num_leaves


def get_number_of_jobs(alidir):
    try:
        num_jobs = int(open('{0}/num_jobs'.format(alidir)).readline().strip())
    except (IOError, ValueError) as e:
        raise Exception("Exception while reading the "
                        "number of alignment jobs: {0}".format(e.errstr))
    return num_jobs


def get_ivector_dim(ivector_dir=None):
    if ivector_dir is None:
        return 0
    stdout_val = get_command_stdout(
        "feat-to-dim --print-args=false "
        "scp:{dir}/ivector_online.scp -".format(dir=ivector_dir))
    ivector_dim = int(stdout_val)
    return ivector_dim

def get_ivector_extractor_id(ivector_dir=None):
    if ivector_dir is None:
        return None
    stdout_val = get_command_stdout(
        "steps/nnet2/get_ivector_id.sh {dir}".format(dir=ivector_dir))

    if (stdout_val.strip() == "") or (stdout_val is None):
        return None

    return stdout_val.strip()

def get_feat_dim(feat_dir):
    if feat_dir is None:
        return 0
    stdout_val = get_command_stdout(
        "feat-to-dim --print-args=false "
        "scp:{data}/feats.scp -".format(data=feat_dir))
    feat_dim = int(stdout_val)
    return feat_dim


def get_feat_dim_from_scp(feat_scp):
    stdout_val = get_command_stdout(
        "feat-to-dim --print-args=false "
        "scp:{feat_scp} -".format(feat_scp=feat_scp))
    feat_dim = int(stdout_val)
    return feat_dim


def read_kaldi_matrix(matrix_file):
    try:
        lines = map(lambda x: x.split(), open(matrix_file).readlines())
        first_field = lines[0][0]
        last_field = lines[-1][-1]
        lines[0] = lines[0][1:]
        lines[-1] = lines[-1][:-1]
        if not (first_field == "[" and last_field == "]"):
            raise Exception(
                "Kaldi matrix file has incorrect format, "
                "only text format matrix files can be read by this script")
        for i in range(len(lines)):
            lines[i] = map(lambda x: int(float(x)), lines[i])
        return lines
    except IOError:
        raise Exception("Error while reading the kaldi matrix file "
                        "{0}".format(matrix_file))


def write_kaldi_matrix(output_file, matrix):
    # matrix is a list of lists
    with open(output_file, 'w') as f:
        f.write("[ ")
        num_rows = len(matrix)
        if num_rows == 0:
            raise Exception("Matrix is empty")
        num_cols = len(matrix[0])

        for row_index in range(len(matrix)):
            if num_cols != len(matrix[row_index]):
                raise Exception("All the rows of a matrix are expected to "
                                "have the same length")
            f.write(" ".join(map(lambda x: str(x), matrix[row_index])))
            if row_index != num_rows - 1:
                f.write("\n")
        f.write(" ]")


def force_symlink(file1, file2):
    import errno
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


def compute_lifter_coeffs(lifter, dim):
    coeffs = [0] * dim
    for i in range(0, dim):
        coeffs[i] = 1.0 + 0.5 * lifter * math.sin(math.pi * i / float(lifter))

    return coeffs


def compute_idct_matrix(K, N, cepstral_lifter=0):
    matrix = [[0] * K for i in range(N)]
    # normalizer for X_0
    normalizer = math.sqrt(1.0 / float(N))
    for j in range(0, N):
        matrix[j][0] = normalizer
    # normalizer for other elements
    normalizer = math.sqrt(2.0 / float(N))
    for k in range(1, K):
        for n in range(0, N):
            matrix[n][
                k] = normalizer * math.cos(math.pi / float(N) * (n + 0.5) * k)

    if cepstral_lifter != 0:
        lifter_coeffs = compute_lifter_coeffs(cepstral_lifter, K)
        for k in range(0, K):
            for n in range(0, N):
                matrix[n][k] = matrix[n][k] / lifter_coeffs[k]

    return matrix


def write_idct_matrix(feat_dim, cepstral_lifter, file_path):
    # generate the IDCT matrix and write to the file
    idct_matrix = compute_idct_matrix(feat_dim, feat_dim, cepstral_lifter)
    # append a zero column to the matrix, this is the bias of the fixed affine
    # component
    for k in range(0, feat_dim):
        idct_matrix[k].append(0)
    write_kaldi_matrix(file_path, idct_matrix)

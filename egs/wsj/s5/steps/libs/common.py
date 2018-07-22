

# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
#           2017 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

""" This module contains several utility functions and classes that are
commonly used in many kaldi python scripts.
"""

from __future__ import print_function
import argparse
import logging
import math
import os
import subprocess
import sys
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


class smart_open(object):
    """
    This class is designed to be used with the "with" construct in python
    to open files. It is similar to the python open() function, but
    treats the input "-" specially to return either sys.stdout or sys.stdin
    depending on whether the mode is "w" or "r".

    e.g.: with smart_open(filename, 'w') as fh:
            print ("foo", file=fh)
    """
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        assert self.mode == "w" or self.mode == "r"

    def __enter__(self):
        if self.filename == "-" and self.mode == "w":
            self.file_handle = sys.stdout
        elif self.filename == "-" and self.mode == "r":
            self.file_handle = sys.stdin
        else:
            self.file_handle = open(self.filename, self.mode)
        return self.file_handle

    def __exit__(self, *args):
        if self.filename != "-":
            self.file_handle.close()


class smart_open(object):
    """
    This class is designed to be used with the "with" construct in python
    to open files. It is similar to the python open() function, but
    treats the input "-" specially to return either sys.stdout or sys.stdin
    depending on whether the mode is "w" or "r".

    e.g.: with smart_open(filename, 'w') as fh:
            print ("foo", file=fh)
    """
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        assert self.mode == "w" or self.mode == "r"

    def __enter__(self):
        if self.filename == "-" and self.mode == "w":
            self.file_handle = sys.stdout
        elif self.filename == "-" and self.mode == "r":
            self.file_handle = sys.stdin
        else:
            self.file_handle = open(self.filename, self.mode)
        return self.file_handle

    def __exit__(self, *args):
        if self.filename != "-":
            self.file_handle.close()


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
        logger.error("Exception while reading the "
                     "number of alignment jobs: ", exc_info=True)
        raise SystemExit(1)
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
    """This function reads a kaldi matrix stored in text format from
    'matrix_file' and stores it as a list of rows, where each row is a list.
    """
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
    """This function writes the matrix stored as a list of lists
    into 'output_file' in kaldi matrix text format.
    """
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


def write_matrix_ascii(file_or_fd, mat, key=None):
    """This function writes the matrix 'mat' stored as a list of lists
    in kaldi matrix text format.
    The destination can be a file or an opened file descriptor.
    If key is provided, then matrix is written to an archive with the 'key'
    as the index field.
    """
    try:
        fd = open(file_or_fd, 'w')
    except TypeError:
        # 'file_or_fd' is opened file descriptor,
        fd = file_or_fd

    try:
        if key is not None:
            print ("{0} [".format(key),
                   file=fd)  # ark-files have keys (utterance-id)
        else:
            print (" [", file=fd)

        num_cols = 0
        for i, row in enumerate(mat):
            line = ' '.join(["{0:f}".format(x) for x in row])
            if i == 0:
                num_cols = len(row)
            elif len(row) != num_cols:
                raise Exception("All the rows of a matrix are expected to "
                                "have the same length")

            if i == len(mat) - 1:
                line += " ]"
            print (line, file=fd)
    finally:
        if fd is not file_or_fd : fd.close()


def read_matrix_ascii(file_or_fd):
    """This function reads a matrix in kaldi matrix text format
    and stores it as a list of lists.
    The input can be a file or an opened file descriptor.
    """
    try:
        fd = open(file_or_fd, 'r')
        fname = file_or_fd
    except TypeError:
        # 'file_or_fd' is opened file descriptor,
        fd = file_or_fd
        fname = file_or_fd.name

    first = fd.read(2)
    if first != ' [':
        logger.error(
            "Kaldi matrix file %s has incorrect format, "
            "only text format matrix files can be read by this script",
            fname)
        raise RuntimeError

    rows = []
    while True:
        line = fd.readline()
        if len(line) == 0:
            logger.error("Kaldi matrix file %s has incorrect format; "
                         "got EOF before end of matrix", fname)
        if len(line.strip()) == 0 : continue # skip empty line
        arr = line.strip().split()
        if arr[-1] != ']':
            rows.append([float(x) for x in arr])  # not last line
        else:
            rows.append([float(x) for x in arr[:-1]])  # lastline
            return rows
    if fd is not file_or_fd:
        fd.close()


def read_key(fd):
  """ [str] = read_key(fd)
   Read the utterance-key from the opened ark/stream descriptor 'fd'.
  """
  str_ = ''
  while True:
    char = fd.read(1)
    if char == '':
        break
    if char == ' ':
        break
    str_ += char
  str_ = str_.strip()
  if str_ == '':
      return None   # end of file,
  return str_


def read_mat_ark(file_or_fd):
    """This function reads a kaldi matrix archive in text format
    and yields a dictionary output indexed by the key (utterance-id).
    The input can be a file or an opened file descriptor.

    Example usage:
    mat_dict = { key: mat for key, mat in read_mat_ark(file) }
    """
    try:
        fd = open(file_or_fd, 'r')
        fname = file_or_fd
    except TypeError:
        # 'file_or_fd' is opened file descriptor,
        fd = file_or_fd
        fname = file_or_fd.name

    try:
        key = read_key(fd)
        while key:
          mat = read_matrix_ascii(fd)
          yield key, mat
          key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


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

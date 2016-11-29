

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module contains various exceptions and errors that are
raised in various kaldi scripts and modules.
"""


class KeyNotUniqueError(Exception):
    """Exception raised when a duplicate key is found.
    """
    def __init__(self, key, location):
        Exception.__init__(self, "Duplicate entry found for key {0} "
                                 "in {1}".format(key, location))


class InputError(Exception):
    """Exception raised when an invalid input is read.
    """
    def __init__(self, err, line=None, input_file=""):
        if line is None:
            Exception.__init__(self, err)
        else:
            Exception.__init__(self, "{err}\nInvalid line {line} in file "
                                     "{f}".format(err=err, line=line,
                                                  f=input_file))


class ArgumentError(Exception):
    """Exception raised when an invalid set of arguments is passed.
    """
    def __init__(self, err):
        Exception.__init__(self, err)


class EmptyObjectError(Exception):
    def __init__(self, err, file_=None):
        if file_ is None:
            Exception.__init__(self, err)
        else:
            Exception.__init__(self, "{err}\nError with file {file}".format(
                err=err, file=file_))

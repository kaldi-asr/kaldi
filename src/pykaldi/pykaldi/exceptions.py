from __future__ import unicode_literals
from pykaldi import __version__, __git_revision__


class PyKaldiError(Exception):
    def __str__(self):
        return 'Pykaldi %s, Git revision %s' % (__version__, __git_revision__)


class PyKaldiCError(PyKaldiError):
    def __init__(self, retcode):
        self.retcode = retcode

    def __str__(self):
        ver = super(PyKaldiError, self).__str__()
        return '%s\nFailed with return code: %s' % repr(ver, self.retcode)


class PyKaldiInstallError(PyKaldiCError):
    pass

[![Build Status](https://travis-ci.org/kaldi-asr/kaldi.svg?branch=master)]
(https://travis-ci.org/kaldi-asr/kaldi)

Kaldi Speech Recognition Toolkit
================================

To build the toolkit: see `./INSTALL`.  These instructions are valid for UNIX
systems including various flavors of Linux; Darwin; and Cygwin (has not been
tested on more "exotic" varieties of UNIX).  For Windows installation
instructions (excluding Cygwin), see `windows/INSTALL`.

To run the example system builds, see `egs/README.txt`

If you encounter problems (and you probably will), please do not hesitate to
contact the developers (see below). In addition to specific questions, please
let us know if there are specific aspects of the project that you feel could be
improved, that you find confusing, etc., and which missing features you most
wish it had.

Kaldi information channels
--------------------------

For HOT news about Kaldi see [the project site](http://kaldi-asr.org/).

[Documentation of Kaldi](http://kaldi-asr.org/doc/):
- Info about the project, description of techniques, tutorial for C++ coding.
- Doxygen reference of the C++ code.

[Kaldi forums and mailing lists](http://kaldi-asr.org/forums.html):
- User list kaldi-help:
    [Web interface/archive](https://groups.google.com/forum/#!forum/kaldi-help) ||
    [Subscribe] (mailto:kaldi-help+subscribe@googlegroups.com) ||
    [Post] (mailto:kaldi-help@googlegroups.com)
- Developer list kaldi-developers:
    [Web interface/archive](https://groups.google.com/forum/#!forum/kaldi-developers) ||
    [Subscribe] (mailto:kaldi-developers+subscribe@googlegroups.com) ||
    [Post] (mailto:kaldi-developers@googlegroups.com)
- Also try luck and search in [SourceForge archives](https://sourceforge.net/p/kaldi/discussion/).

Development pattern for contributors
------------------------------------

1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/)
   of the [main Kaldi repository] (https://github.com/kaldi-asr/kaldi) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create
   a branch `my-awesome-feature`.
3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/)
   through the Web interface of GitHub.
4. As a general rule, please follow [Google C++ Style Guide]
   (https://google-styleguide.googlecode.com/svn/trunk/cppguide.html).
   There are a [few exceptions in Kaldi](http://kaldi-asr.org/doc/style.html).
   You can use the [Google's cpplint.py]
   (https://google-styleguide.googlecode.com/svn/trunk/cpplint/cpplint.py)
   to verify that your code is free of basic mistakes.

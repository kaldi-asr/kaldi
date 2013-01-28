
This directory contains various things: the source and pdf for some
Kaldi-related papers in papers/ (this does not contain all Kaldi-related papers),
the Kaldi logo in logo, and some conversion scripts for converting HTK to Kaldi 
models.

WARNING: the HTK conversion scripts may not work; bugs have been reported.  In
general, conversion back and forth between HTK and Kaldi is not something we
recommend doing or something that we aim to support in future.  The problem is
that HTK and Kaldi have so many differences, both large differences in
philosophy and small differences in details, that conversion in general is not
possible, and even where possible, it is probably not a good idea.  If someone
else wants to keep these scripts updated, we would appreciate that, but the core
Kaldi maintainers do not view this as a good use of their time.

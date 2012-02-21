
This directory contains example scripts that demonstrate how to 
use Kaldi.  Each subdirectory corresponds to a corpus that we have
example scripts for.  Currently these are both corpora available from
the Linguistic Data Consortium (LDC).

Explanations of the corpora are below:

 wsj: The Wall Street Journal corpus.  This is a corpus of read
    sentences from the Wall Street Journal, recorded under clean conditions.
    The vocabulary is quite large. 
    Available from the LDC as either: [ catalog numbers LDC93S6A (WSJ0) and LDC94S13A (WSJ1) ]
    or: [ catalog numbers LDC93S6B (WSJ0) and LDC94S13B (WSJ1) ]
    The latter option is cheaper and includes only the Sennheiser
    microphone data (which is all we use in the example scripts).

 rm: Resource Management.  Clean speech in a medium-vocabulary task consisting
    of commands to a (presumably imaginary) computer system.
    Available from the LDC as catalog number LDC93S3A (it may be possible to
    get the same data using combinations of other catalog numbers, but this
    is the one we used).

Recipes in progress:

 swbd: Switchboard.  A fairly large amount of telephone speech (2-channel, 8kHz
    sampling rate).
    This directory is a work in progress.
  

 gp: GlobalPhone.  This is a multilingual speech corpus.

 timit: TIMIT, which is an old corpus of carefully read speech.  

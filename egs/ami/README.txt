
About the AMI corpus:

WEB: http://groups.inf.ed.ac.uk/ami/corpus/
LICENCE: http://groups.inf.ed.ac.uk/ami/corpus/license.shtml

"The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals synchronized to a common timeline. These include close-talking and far-field microphones, individual and room-view video cameras, and output from a slide projector and an electronic whiteboard. During the meetings, the participants also have unsynchronized pens available to them that record what is written. The meetings were recorded in English using three different rooms with different acoustic properties, and include mostly non-native speakers." See http://groups.inf.ed.ac.uk/ami/corpus/overview.shtml for more details.


About the recipe:

s5)

The scripts under this directory build systems using AMI data only, this includes both training, development and evaluation sets (following Full ASR split on http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml). This is different from RT evaluation campaigns that usually combined couple of different meeting datasets from multiple sources. In general, the recipe reproduce baseline systems build in [1] but without propeirary components* that means we use CMUDict [2] and in the future will try to use open texts to estimate background language model.

Currently, one can build the systems for close-talking scenario, for which we refer as
-- IHM (Individual Headset Microphones)
and two variants of distant speech
-- SDM (Single Distant Microphone) using 1st micarray and,
-- MDM (Multiple Distant Microphones) where the mics are combined using BeamformIt [3] toolkit.

To run all su-recipes the following (non-standard) software is expected to be installed
1) SRILM - to build language models (look at KALDI_ROOT/tools/install_srilm.sh)
2) BeamformIt (for MDM scenario, installed with Kaldi tools)
3) Java (optional, but if available will be used to extract transcripts from XML)

[1] "Hybrid acoustic models for distant and multichannel large vocabulary speech recognition", Pawel Swietojanski, Arnab Ghoshal and Steve Renals, In Proc. ASRU, December 2013
[2] http://www.speech.cs.cmu.edu/cgi-bin/cmudict
[3] "Acoustic beamforming for speaker diarization of meetings", Xavier Anguera, Chuck Wooters and Javier Hernando, IEEE Transactions on Audio, Speech and Language Processing, September 2007, volume 15, number 7, pp.2011-2023.

*) there is still optional dependency on Fisher transcripts (LDC2004T19, LDC2005T19) to build background language model and closely reproduce [1].


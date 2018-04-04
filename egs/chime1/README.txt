
This is a kaldi setup for 1st CHiME challenge. See
http://spandh.dcs.shef.ac.uk/projects/chime/challenge.html
for more detailed information.

The setup should also work for GRID corpus and 2nd CHiME challenge track 1
http://spandh.dcs.shef.ac.uk/gridcorpus/
http://spandh.dcs.shef.ac.uk/chime_challenge/chime2013/


Quick instruction:

1) download CHiME1 data

Check the download page http://spandh.dcs.shef.ac.uk/projects/chime/PCC/datasets.html
Train set
http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_train_reverberated.tar.gz
Devel set
http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_devel_isolated.tar.gz
Test set
http://spandh.dcs.shef.ac.uk/projects/chime/PCC/data/PCCdata16kHz_test_isolated.tar.gz

2) move to Kaldi CHiME1 directory, e.g.,

cd kaldi-trunk/egs/chime1/s5

3a) specify Kaldi directory in path.sh,

export KALDI_ROOT="<your kaldi directory>/kaldi-trunk"

3b) specify CHiME1 signal directory and CHiME1 recogniser directory for your
username ($USER) in config.sh.

By default, directories data/ exp/ mfcc/ will be created by the recipe in the
Kaldi CHiME1 recogniser directory. You could link these to directories on a 
different disk space or specify a different directory in config.sh,

export WAV_ROOT="<your CHiME1 directory>/PCCdata16kHz"
export REC_ROOT="."

4) execute run.sh

./run.sh

4*) we suggest to use the following command to save the main log file

nohup ./run.sh > run.log

5) You can find result at exp/tri2b/decode_*/keyword_scores.txt


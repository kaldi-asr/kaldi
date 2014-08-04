Summary
-------
The data comprise over 41 hours of speech in English.

The English recordings were collected from humans interacting via telephone 
calls with statistical dialogue systems, designed to provide the user 
with information on a suitable dining venue in the town.

The data collection process is described in detail
in article "Free English and Czech telephone speech corpus shared under the CC-BY-SA 3.0 license"
published for LREC 2014 (To Appear).

WE USE COMMON KALDI DECODERS IN THE SCRIPTS (gmm-latgen-faster through steps/decode.sh)
However, the main purpose of providing the data and scripts
is training acoustic models for real-time speech recognition unit
for dialog system ALEX, which uses modified real-time Kaldi OnlineLatgenRecogniser.
The modified Kaldi decoders are NOT required for running the scripts!

The modified OnlineLatgenRecogniser is actively developed at 
https://github.com/UFAL-DSG/pykaldi/tree/master/src/onl-rec
and has Python wrapper:
https://github.com/UFAL-DSG/pykaldi/tree/master/src/pykaldi
Note that I am currently moving the online recogniser to:
http://sourceforge.net/p/kaldi/code/HEAD/tree/sandbox/oplatek2/

Credits and license
------------------------
The scripts are partially based on Voxforge KALDI recipe.
The original scripts as well as theses scripts are licensed under APACHE 2.0 license.
The data are distributed under Attribution-{ShareAlike} 3.0 Unported ({CC} {BY}-{SA} 3.0) license.
Czech data: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-4670-6
English data: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-4671-4

The data collecting process and development of these training scripts 
was partly funded by the Ministry of Education, Youth and Sports 
of the Czech Republic under the grant agreement LK11221 
and core research funding of Charles University in Prague.
For citing, please use following BibTex citation:

@inproceedings{korvas_2014,
  title={{Free English and Czech telephone speech corpus shared under the CC-BY-SA 3.0 license}},
  author={Korvas, Mat\v{e}j and Pl\'{a}tek, Ond\v{r}ej and Du\v{s}ek, Ond\v{r}ej and \v{Z}ilka, Luk\'{a}\v{s} and Jur\v{c}\'{i}\v{c}ek, Filip},
  booktitle={Proceedings of the Eigth International Conference on Language Resources and Evaluation (LREC 2014)},
  pages={To Appear},
  year={2014},
}


Expected results
----------------
The expected results were obtained simply by running
bash train_voip_cs.sh OR bash train_voip_en.sh.
Note that you need SRILM installed in path or at kaldi/tools/ directory!
See s5/RESULTS. Following notation is used:

    build2 - bigram LM from train data, estimated by the scripts using SRILM
    build0 - zerogram LM from test data, estimated by scripts using Python code
    LMW - Language model weight, we picked the best from (min_lmw, max_lmw) based on decoding results on DEV set


Details
-------
* Requires Kaldi installation and Linux environment. (Tested on Ubuntu 10.04, 12.04 and 12.10.)
* The config file s5/env_voip_en.sh sets the data directory,
  mfcc directory and experiments directory.
* Our scripts prepare the data to the expected format in s5/data.
* Experiment files are stored to $exp directory e.g. s5/exp.
* The local directory contains scripts for data preparation to prepare 
  lang directory.
* path.sh, cmd.sh and  common/* contain configurations for the 
  recipe.
* Language model (LM) is either built from the training data using 
  [SRILM](http://www.speech.sri.com/projects/srilm/)  or we supply one in 
  the ARPA format.


Running experiments
-------------------
Before running the experiments, check that:

* you have the Kaldi toolkit compiled: 
  http://sourceforge.net/projects/kaldi/.
* you have SRILM compiled. (This is needed for building a language model 
  unless you supply your own LM in the ARPA format.) 
  See http://www.speech.sri.com/projects/srilm/.
* The number of jobs njobs are set correctly in path.sh.
* In cmd.sh, you switched to run the training on a SGE[*] grid if 
  required (disabled by default).

Start the recipe from the s5 directory by running 
bash run.sh.
It will create s5/mfcc, s5/data and s5/exp directories.
If any of them exists, it will ask you if you want them to be overwritten.

.. [*] Sun Grid Engine

Extracting the results and trained models
-----------------------------------------
The main scripts, s5/run.sh, 
perform not only training of the acoustic models, but also decoding.
The acoustic models are evaluated after running the training and  
reports are printed to the standard output.

The s5/local/results.py exp command extracts the results from the $exp directory.
and stores the results to exp/results.log.

If you want to use the trained acoustic model with your language model
outside the prepared script, you need to build the HCLG decoding graph yourself.  
See http://kaldi.sourceforge.net/graph.html for general introduction to the FST 
framework in Kaldi.

The simplest way to start decoding is to use the same LM which
was used by the s5/run.sh script.
Let's say you want to decode with 
the acoustic model stored in exp/tri2b_bmmi,
then you need files listed below:

================================= =====================================================================================
mfcc.conf                          Speech parametrisation (MFCC) settings. Training and decoding setup must match.
exp/tri2b_bmmi/graph/HCLG.fst      Decoding Graph. Graph part of AM plus lexicon, phone->3phone & LM representation.
exp/tri2b_bmmi/graph/words.txt     Word symbol table, a mapping between words and integers which are decoded.
exp/tri2b_bmmi/graph/silence.csl   List of phone integer ids, which represent silent phones. 
exp/tri2b_bmmi/final.mdl           Trained acoustic model (AM).
exp/tri2b_bmmi/final.mat           Trained matrix of feature/space transformations (E.g. LDA and bMMI).
================================= =====================================================================================


We recommend to study steps/decode.sh Kaldi standard script
for standalone decoding with gmm-latgen-faster Kaldi decoder.

In order to build your own decoding graph HCLG 
you need LM in ARPA format and files in table below. 

* Note 1: Building HCLG decoding graph is out of scope this README.
* Note 2: Each acoustic model needs corresponding HCLG graph.
* Note 3: The phonetic dictionary applied on the vocabulary 
  should always generate only a subset of phones seen in training data!

===============================  =========================================================================
LM.arpa                           Language model in ARPA format [You should supply it]
vocabulary.txt                    List of words you want to decode [You should supply it]
OOV_SYMBOL                        String representing out of vocabulary word. [You should supply it]
dictionary.txt                    Phonetic dictionary. [You should supply it]
exp/tri2b_bmmi/final.mdl          Trained acoustic model (AM).
exp/tri2b_bmmi/final.tree         Phonetic decision tree.
===============================  =========================================================================

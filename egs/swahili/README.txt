### 
# Swahili Data collected by Hadrien Gelas 
# Prepared by Tien-Ping Tan & Laurent Besacier
# GETALP LIG, Grenoble, France
###


### OVERVIEW
The package contains swahili speech corpus with audio data in the directory asr_swahili/data. The data directory contains 2 subdirectories:
a. train - speech data and transription for training automatic speech recognition system (Kaldi ASR format [1])
b. test - speech data and transription for testing automatic speech recognition system (Kaldi ASR format)

A text corpus and language model in the directory asr_swahili/LM, and lexicon in the directory asr_swahili/lang

### PUBLICATION ON SWAHILI SPEECH & LM DATA
More details on the corpus and how it was collected can be found on the following publication (please cite this bibtex if you use this data)

 @InProceedings { gelas:hal-00954048,
  author = {Gelas, Hadrien and Besacier, Laurent and Pellegrino, Francois},
  title = {{D}evelopments of {S}wahili resources for an automatic speech recognition system},
  booktitle = {{SLTU} - {W}orkshop on {S}poken {L}anguage {T}echnologies for {U}nder-{R}esourced {L}anguages},
  year = {2012},
  address = {Cape-Town, Afrique Du Sud},
  abstract = {no abstract},
  x-international-audience = {yes},
  url = {http://hal.inria.fr/hal-00954048},
}

### SWAHILI SPEECH CORPUS
Directory: asr_swahili/data/train
Files: text (training transcription), wav.scp (file id and path), utt2spk (file id and audio id), spk2utt (audio id and file id), wav (.wav files). 
For more information about the format, please refer to Kaldi website http://kaldi-asr.org/doc/data_prep.html
Description: training data in Kaldi format about 10 hours. Note: The path of wav files in wav.scp have to be modified to point to the actual locatiion.  

Directory: asr_swahili/data/test
Files: text (test transcription), wav.scp (file id and path), utt2spk (file id and audio id), spk2utt (audio id and file id), wav (.wav files)
Description: testing data in Kaldi format about 1.8 hours. The audio files for testing has the format 
ID_16k-emission_swahili_TThTT_-_TThTT_tu_YYYYMMDD_part001Q, where ID is the audio id, T is the time, Y is the year, M is the month and D is the day of recording. The 
last character Q indicate the quality of the utterance. g is good, m is utterance with background music, n is utterance with noise, s is utterance
with overlap speech, l is very noise utterance. Note: The path of wav files in wav.scp have to be modified to point to the actual locatiion. 



### SWAHILI TEXT CORPUS
Directory: asr_swahili/LM
Files: 00-LM_SWH-CORPUS.txt, 01-CLN4-TRN.txt.zip, 02-CLN4-DEV.txt and 03-CLN4-TST.txt, swahili.arpa.zip

N.B.: You need to unzip 01-CLN4-TRN.txt.zip for 01-CLN4-TRN.txt and swahili.arpa.zip for swahili.arpa

# /00-LM_SWH-CORPUS.txt
Contains 28 M Words. Full text data grabbed from online newspaper and cleaned as much as it could
All files below are extracted from this file

# /01-CLN4-TRN.txt
Training data for LM

# /02-CLN4-DEV.txt
Dev data for LM

# /03-CLN4-TST.txt
Testing data for LM

# /swahili.arpa
A language model created using SRILM [2] using the text from the text in 01-CLN4-TRN.txt



### LEXICON/PRONUNCIATION DICTIONARY
Directory: asr_swahili/lang
Files: lexicon.txt (lexicon), nonsilence_phones.txt (speech phones), optional_silence.txt (silence phone)
Description: lexicon contains words and their respective pronunciation, non-speech sound and noise in Kaldi format. G2P conversion rules, please refer to [3]

### SCRIPTS
In asr_swahili/kaldi-scripts you find the scripts used to train and test models
from the existing data and lang directory you find scripts for run the sequence one by one : 04_train_mono.sh + 04a_train_triphone.sh + 04b_train_MLLT_LDA.sh + 04c_train_SAT_FMLLR.sh + 04d_train_MMI_FMMI.sh + 04e_train_sgmm.sh

### RUN.SH
This script automatically creates all you need to build an ASR for swahili with Kaldi : it prepares the data and runs the models. 
At the end, you should obtain the results below.

### WER RESULTS OBTAINED SO FAR (you should obtain the same on this data if same protocol used)
Monophone (13 MFCC): 49.28% (All)
Triphone (13 MFCC): 33.55% (All)
Triphone (13 MFCC + delta + delta2): 33.61% (All)
Triphone (39 features) + LDA and MLLT: 31.92% (All)
Triphone (39 features) + LDA and MLLT + SAT and FMLLR: 31.56% (All)
Triphone (39 features) + LDA and MLLT + SAT and FMLLR + MMI and fMMI: 30.87% (All)
Triphone (39 features) + LDA and MLLT + SGMM: 27.36% (All)


### REFERENCES
[1] KALDI: http://kaldi-asr.org/doc/tutorial_running.html
[2] SRILM: http://www.speech.sri.com/projects/srilm/
[3] Hadrien Gelas, Laurent Besacier, François Pellegrino, Developments of Swahili Resources for an Automatic Speech Recognition System, 
http://www.ddl.ish-lyon.cnrs.fr/fulltext/Gelas/Gelas_2012_SLTU.pdf 

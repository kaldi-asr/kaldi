# Api.ai model decoding example scripts
This directory contains scripts on how to use a pre-trained chain english model and kaldi base code to recognize any number of wav files.

IMPORTANT: wav files must be in 16kHz, 16 bit little-endian format.

## Model
English pretrained model were released by Api.ai under Creative Commons Attribution-ShareAlike 4.0 International Public License. 
- Acoustic data is mostly mobile recorded data
- Language model is based on Assistant.ai logs and good for understanding short commands, like "Wake me up at 7 am"
For more details, visit https://github.com/api-ai/api-ai-english-asr-model

## Usage
Ensure kaldi is compiled and this scripts are inside kaldi/egs/<subfolder>/ directory then run
```sh
$ ./download-model.sh # to download pretrained chain model
$ ./recognize-wav.sh test1.wav test2.wav # to do recognition
```
See console output for recognition results.

### Using steps/nnet3/decode.sh
You can use kaldi steps/nnet3/decode.sh, which will decode data and calculate Word Error Rate (WER) for it.

Run:
```sh
$ recognize-wav.sh test1.wav test2.wav
```
It will make data dir, calculate mfcc features for it and do decoding, you need only first two steps out of it. If you want WER then edit data/test-corpus/text and replace NO_TRANSCRIPTION with expected text transcription for every wav file. 

Run for decoding:
```sh
$ steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --cmd run.pl --nj 1 exp/api.ai-model/ data/test-corpus/ exp/api.ai-model/decode/
```
See exp/api.ai-model/decode/wer* files for WER and exp/api.ai-model/decode/log/ files for decoding output.

### Online Decoder:
See http://kaldi-asr.org/doc/online_decoding.html for more information about kaldi online decoding.

Run:
```sh
$./local/create-corpus.sh data/test-corpus/ test1.wav test2.wav
```
If you want WER then edit data/test-corpus/text and replace NO_TRANSCRIPTION with expected text transcription for every wav file.

Make config file exp/api.ai-model/conf/online.conf with following content:
```
--feature-type=mfcc
--mfcc-config=exp/api.ai-model/mfcc.conf
```
Then run:
```sh
$ steps/online/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --cmd run.pl --nj 1 exp/api.ai-model/ data/test-corpus/ exp/api.ai-model/decode/
```
See exp/api.ai-model/decode/wer* files for WER and exp/api.ai-model/decode/log/ files for decoding output.

This is a baseline Kaldi recipe for the LibriCSS data. For the official
data and code, check out [the official repo](https://github.com/chenzhuo1011/libri_css).

This recipe addresses the problem of speech recognition in a meeting-like
scenario, where multiple overlapping speakers may be present, and the 
number of speakers is not known beforehand. For details of the LibriCSS
data, check out the paper on [ArXiv](https://arxiv.org/abs/2001.11482)

We provide examples for monoaural (1 channel) and 7-channel cases. Each of
these is present in the respective directories. The pipeline is based on
the baseline recipe for the CHiME-6 challenge. 

For ease of reproduction, we include the training for all modules in the
recipe. However, the diarization and ASR pre-trained models are also available 
from kaldi-asr.org/models:

* Speaker diarization: CHiME-6 baseline x-vector + AHC diarizer, trained on VoxCeleb 
with simulated RIRs available [here](http://kaldi-asr.org/models/m12).
* ASR: Chain model trained on 960h clean Librispeech training data available
[here](http://kaldi-asr.org/models/m13). 

The SAD pre-trained model is not publicly available. We train a SAD on a 300h 
subset of the Librispeech data.
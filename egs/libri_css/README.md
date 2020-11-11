### LibriCSS integrated recipe

This is a Kaldi recipe for the LibriCSS data, providing diarization and
ASR on mixed single-channel and separated audio inputs. 

#### Data
We use the LibriCSS data released with the following paper:
```
@article{Chen2020ContinuousSS,
  title={Continuous Speech Separation: Dataset and Analysis},
  author={Z. Chen and T. Yoshioka and Liang Lu and T. Zhou and Zhong Meng and Yi Luo and J. Wu and J. Li},
  journal={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2020}
}
```
For the official data and code, check out [the official repo](https://github.com/chenzhuo1011/libri_css).

#### Recipe details
This recipe addresses the problem of speech recognition in a meeting-like
scenario, where multiple overlapping speakers may be present, and the 
number of speakers is not known beforehand. 

We provide recipes for 2 scenarios:
1. `s5_mono`: This is a single channel diarization + ASR recipe which takes as the
input a long single-channel recording containing mixed audio. It then performs SAD,
diarization, and ASR on it and outputs speaker-attributed transcriptions, 
which are then evaluated with cpWER (similar to CHiME6 Track 2).
2. `s5_css`: This pipeline uses a speech separation module at the beginning,
so the input is 2-3 separated audio streams. We assume that the separation is
window-based, so that the same speaker may be split across different streams in
different windows, thus making diarization necessary.

#### Pretrained models for diarization and ASR
For ease of reproduction, we include the training for both modules in the
recipe. We also provide pretrained models for both diarization and ASR 
systems.

* SAD: CHiME-6 baseline TDNN-Stats SAD available [here](http://kaldi-asr.org/models/m12).
* Speaker diarization: CHiME-6 baseline x-vector + AHC diarizer, trained on VoxCeleb 
with simulated RIRs available [here](http://kaldi-asr.org/models/m12).
* ASR: We used the chain model trained on 960h clean LibriSpeech training data available
[here](http://kaldi-asr.org/models/m13). It was then additionally fine-tuned for 1
epoch on LibriSpeech + simulated RIRs. For LM, we trained a TDNN-LSTM language model
for rescoring. All of these models are available at this 
[Google Drive link](https://drive.google.com/file/d/13ceXdK6oAUuUyxn7kjQVVqpe8r6Sc7ds/view?usp=sharing).

#### Speech separation
The speech separation module has not been provided. If you want to use the
`s5_css` recipe, check out [this tutorial](https://desh2608.github.io/pages/jsalt/) for
instructions on how to plug in your component into the pipeline.

If you found this recipe useful for your experiments, consider citing:

```
@article{Raj2021Integration,
  title={Integration of speech separation, diarization, and recognition for multi-speaker meetings:
System description, Comparison, and Analysis},
  author={D.Raj and P.Denisov and Z.Chen and H.Erdogan and Z.Huang and M.He and S.Watanabe and
  J.Du and T.Yoshioka and Y.Luo and N.Kanda and J.Li and S.Wisdom and J.Hershey},
  journal={IEEE Spoken Language Technology Workshop 2021},
  year={2021}
}
```
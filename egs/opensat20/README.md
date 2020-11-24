###  OpenSAT 2020 recipe (https://sat.nist.gov/opensat20#tab_overview)

This is a Kaldi based setup for the SAFE-T data (https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/lrec2020-safe-t-corpus.pdf).
The aim of this setup is to provide example scripts to use out-of-domain data in low
resource condition. It provides three recipes (run.sh, run_shared.sh, run_finetune.sh).
The description of these three setups is as follows:

`Target data acoustic model(run.sh)`: We trained an ASR system using the 40 hours of SAFE-T
training data, and evaluated it on the OpenSAT20 Dev audio. We used CNN-TDNN-F
architecture, which contains 6 CNN layers followed by 9 TDNN-F layers each with 1024
neurons, and bottle-neck factorization to 128 dimensions with stride 3. Speed perturbation
as augmentation is used to increase data size by a factor of 3, in addition online spectral
augmentation is used to make each mini-batch unique and increase robustness of the model.
With this setup, we can quickly train an ASR system and to get a baseline WER. The
resulting WER is available in RESULTS doc.

`Fine tuning (run_finetune.sh)`: To increase the amount of training data for the acoustic
model training, we used AMI and ICSI speech data with speed perturbation as data augmentation.
We used CNN-TDNN-F architecture, which contains 6 CNN layers followed by 9 TDNN-F layers
each with 1536 neurons, and bottle-neck factorization to 160 dimensions with stride 3. Speed
perturbation as augmentation and online spectral augmentation is used with this setup as well.
After training the acoustic model with AMI and ICSI datasets, fine tuning is performed with
a lower learning rate with the SAFE-T dataset. The resulting WER is available in RESULTS doc.

`Shared(run_shared.sh)`: Adding AMI and ICSI data to the SAFE-T data and training an acoustic
model with these three datasets. Similar to the Fine tuning setup, same CNN-TDNN-F
architecture is used with speed perturbation and spectral augmentation. We are adding a script
for other augmentations in this setup. The resulting WER is available in RESULTS doc.

#### Data
We use the SAFE-T data released with the following paper:
```
@inproceedings{delgado2020safe,
  title={The SAFE-T Corpus: A New Resource for Simulated Public Safety Communications},
  author={Delgado, Dana and Walker, Kevin and Strassel, Stephanie and Jones, Karen Sp{\"a}rck and Caruso, Christopher and Graff, David},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={6450--6457},
  year={2020}
}
```

SAFE-T (speech analysis for emergency response technology) has 131 hrs (labelled and unlabelled)
of single-channel 48 kHz training data. Most of the speakers are native English speakers. The 
participants are playing the game of Flashpoint fire rescue. The recordings do not have overlap,
little reverberation but have significant noise. The noise is artificial and the SNR varies with
time. The noise level varies from 0-14db or 70-85 dB. The noises are car ambulances, rain,
or similar sounds. There are a total of 87 speakers.

```

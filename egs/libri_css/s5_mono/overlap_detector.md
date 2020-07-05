## Kaldi-based Overlap Detector

This is a how-to for training/using the Kaldi-based overlap detector. We
assume that Kaldi is already installed.

### Overview

This is an HMM-DNN based overlap detector. A frame-level classifier is trained
to predict: silence, single speaker, and overlap classes. The output of
this classifier is treated as posterior probabilities of the classes, and
used as emission probabilities for an HMM. The graph is dynamically
constructed based on min/max silence/speech constraints. Direct transition
between the silence and overlap states is prohibited.

The classifier is based on a time-delay neural network for long temporal 
context, with statistics pooling. It is based on the Aspire and CHiME-6
speech activity detection models available in Kaldi.

### Training 

The example script trains the overlap detector on the full simulated LibriCSS
training data, containing 18893 recordings (each ranging between 1-3 minutes).
Since the training data is large, we only train it for a single epoch.

To train the overlap detector on your own data, you need to make necessary
modifications in the script `local/train_overlap_detector.sh` (mainly in
stage 0 -- data preparation).

An example decoding graph is also provided here. This graph was constructed
with the following state transition constraints, passed through the `--graph-opts`
parameter in the `local/detect_overlaps.sh` script:

```
  --min-silence-duration=0.03
  --min-speech-duration=0.3
  --max-speech-duration=10.0
  --min-overlap-duration 0.1 
  --max-overlap-duration 4.0
```

We also merge consecutive segments after decoding during post-processing, and do
not apply any collar to extend the detected segments. These parameters can also
be controlled using the following parameters in `local/detect_overlaps.sh`:

```
 --segment_padding=0
 --merge_consecutive_max_dur=inf
```

We haven't tuned these parameters heavily, and it may be possible to improve
detection performance by modifying them.

### Using the pretrained model

For convenience, we provide a pretrained model (which was trained as described
above). To use the model, extract the `tar.gz` file, and copy over the contents
of the `exp` directory to your own `exp` directory. Also copy the contents of
the `local` directory, which contains training and decoding scripts for the 
overlap detector. Then, use the following code to get overlap segments in
your own data directory:

```
local/detect_overlaps.sh data/dev \
  exp/overlap_1a/tdnn_stats_1a exp/overlap_1a/dev
```

After the decoding finishes, the output RTTM is written to the location:
`exp/overlap_1a/dev/rttm_overlap`

Additionally, the overlap detector can also be used to obtain only the single
speaker regions. If you have already run the above steps, use the following:

```
local/detect_overlaps.sh --stage 4 --region-type single data/dev \
  exp/overlap_1a/tdnn_stats_1a exp/overlap_1a/dev
```

Otherwise, run the above without the `--stage 4` option. 

An important parameter that might need to be tuned to balance the miss speech 
vs. false alarm is the `--output-scale`. It should be of the form 
`<silence-scale> <single-scale> <overlap-scale>`. The results below (on the 
real LibriCSS data) were obtained with the setting `--output-scale "1 1 10"`.

### Performance

The pretrained model has the following performance on the simulated dev and test
sets (measured against a force-aligned RTTM):

|      | Missed speech | False alarm |
|------|---------------|-------------|
| Dev  | 2.6%          | 1.1%        |
| Test | 2.0%          | 1.2%        |

On the actual LibriCSS dev and eval sets, it performs as follows (measured
against the annotation RTTM).

|      | Missed speech | False alarm |
|------|---------------|-------------|
| Dev  | 5.9%          | 3.8%        |
| Eval | 5.4%          | 4.1%        |

### Contact

This overlap detector was implemented and trained by Desh Raj during the
JSALT 2020 workshop at Johns Hopkins University. If you run into any
issues with the code, contact me at: r.desh26@gmail.com
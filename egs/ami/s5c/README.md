This recipe performs diarization using x-vectors on the AMI mix-headset data.
We demonstrate the use of different clustering methods: AHC, spectral, and
VBx. The recipe also shows how to train a TDNN-LSTM overlap detector using annotation
or force aligned targets, and use it to detect overlapping segments in a
recording.

## Note on reference and split

We used the official `AMI_public_manual_1.6.2` annotations to generate the reference
RTTM files. The annotations contain `word` and `vocal sound` tags, where the latter
contains a wide variety of sounds (the majority being laughter, clapping, etc.).
For the purposes of this "standard evaluation", we ignore all the vocal sounds
and only consider the word annotations. We merged adjacent word segments if there
was no pause between them.

The train/dev/test split is obtained from the official AMI Full Corpus partition.
Note that this is different from the Full Corpus ASR partition used in the s5 and
s5b recipes (which are for ASR).

For more details, please refer to Section 4 in: https://arxiv.org/pdf/2012.14952.pdf

The data split and reference RTTMs are available at:
`https://github.com/BUTSpeechFIT/AMI-diarization-setup`

## Results

We report results below using oracle SAD and no overlap detection. The dev and test
sets contain 13.5% and 14.6% overlaps, respectively, which accounts for missed
speech. False alarm is 0% since oracle SAD is used. DER is computed as the sum
of missed speech (=overlap %), false alarm (=0), and speaker confusion (SE).

| Method   | Dev SE | Dev DER | Test SE | Test DER |
|----------|--------|---------|---------|----------|
| AHC      | 7.2    | 20.7    | 9.7     | 24.3     |
| Spectral | 6.2    | 19.7    | 5.6     | 20.2     |
| VBx      | 6.0    | 19.5    | 8.4     | 23.0     |

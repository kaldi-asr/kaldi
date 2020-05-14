# HIMIA SPEAKER VERIFICATION

This is repository for speaker verification experiments with [HIMIA (openslr-85)](http://www.openslr.org/85/) dataset.

## Usage
One can simply run `./run.sh` to download open-sourced part of data. While there are two special cases:
* If one has the open-sourced part in place, one can run `./run.sh --stage 0` to skip the downloading & preparation part of those data.
* If one has AISHELL2 data, please run `./run.sh --include-aishell2 true --aishell2-root $your_aishell2_folder`.

## RESULTS
Please check `local/run_text_independent.sh` for results, which shall be able to be replicated without AISHELL2 data involved.

## References
There is [a paper](https://arxiv.org/abs/1912.01231) on a more detailed description about HIMIA with some preliminary numbers (different from ones here since they were generated using different framework). If one would like to use HIMIA in experiments, please cite the paper as below:
```
@INPROCEEDINGS{9054423,
  author={X. {Qin} and H. {Bu} and M. {Li}},
  booktitle={Proc. ICASSP 2020}, 
  title={HI-MIA: A Far-Field Text-Dependent Speaker Verification Database and the Baselines}, 
  year={2020},
  volume={},
  number={},
  pages={7609-7613}
}
```

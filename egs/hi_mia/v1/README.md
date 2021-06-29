# HIMIA SPEAKER VERIFICATION

This is repository for speaker verification experiments with [HIMIA (openslr-85)](http://www.openslr.org/85/) dataset. It is primarily a text-dependent dataset (see `../w1` for keyword spotting system) but here for speaker verification, we also include text-independent scenario.

## Usage
One can simply run `./run.sh` to download open-sourced part of data. While there are two special cases:
* If one has the open-sourced part in place, one can run `./run.sh --stage 0` to skip the downloading & preparation part of those data. By default the data will be downloaded to/stored in `$pwd/corpora/`. If the data has been downloaded somewhere else or seperately, please combine them together (by calling `utils/combine_data.sh $target_data_path $data_paths` for example) into somewhere and run `./run.sh --stage 0 --corporadir $your_data_path`.
* If one has AISHELL2 data, please run `./run.sh --stage $stage --include-aishell2 true --aishell2-root $your_aishell2_folder`. Again, `$stage` can be set to -10 (default) if one does not have multi_cn data and -1 otherwise.
* If one would like to apply for AISHELL2 data, please check `../../aishell2/README.md` on how to do it.

## RESULTS
Please check `local/run_text_independent.sh` for text-independent system results, which shall be able to be replicated with or without AISHELL2 data involved. Text-dependent settings are yet to be developed.

## References
There is [a paper](https://arxiv.org/abs/1912.01231) on a more detailed description about HIMIA with some preliminary numbers (different from ones here since they were generated using different framework). If one would like to use HIMIA in experiments, please cite the paper as below:
```
@INPROCEEDINGS{himia_data,
  author={X. {Qin} and H. {Bu} and M. {Li}},
  booktitle={Proc. ICASSP 2020}, 
  title={HI-MIA: A Far-Field Text-Dependent Speaker Verification Database and the Baselines}, 
  year={2020},
  pages={7609-7613}
}
```

## TODO
[ ] [Text-dependent setting]

## Contact
If having any problem, please open pull request and @underdogliu

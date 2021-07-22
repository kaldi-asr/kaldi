# LT-LM: a novel non-autoregressive language model for single-shot lattice rescoring
[Paper](https://arxiv.org/pdf/2104.02526.pdf)

## Setup:
`cd fairseq_ltlm && setup.sh`
## run:
* put slurm.conf to conf/
* modify fairseq\_ltlm/recipes/config.sh if needed
* `bash fairseq\_ltlm/recipes/run.sh`
## evaluate:
For evaluation, you can 
run fairseq\_ltlm/recipes/run\_5\_eval.sh (see run.sh) or use fairseq\_ltlm/ltlm/eval.py directly.

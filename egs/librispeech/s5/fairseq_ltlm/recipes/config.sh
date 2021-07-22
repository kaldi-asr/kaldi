#!/bin/bash
########################################################################
####### Config for LTLM based on kaldi libri recipe #####
########################################################################

################################
#### infractructure config #####
################################
cmd_cpu='utils/slurm.pl --max-jobs-run 300 --config conf/slurm_cpu.conf'
cmd_gpu='utils/slurm.pl --max-jobs-run 12  --config conf/slurm_gpu.conf'
train_cmd=$cmd_gpu
decode_nj=30

################################
#### Input data Parameters #####
################################
basedir=data
acoustic_train_dirs="data/train_960_cleaned_hires"			
								# Data with wavs.
test_dirs="$basedir/dev_clean_hires $basedir/dev_other_hires $basedir/test_clean_hires $basedir/test_other_hires"
								# Subsets for testing.
lang=$basedir/lang_chain
								# Any lang. Prefer decode lang.
filter=cat 
								# wer_filter, which aplied before scoring.
graph=exp/chain_cleaned/tdnn_1d_sp/graph_tgsmall
								# Standard decode graph.

#################################
########### AM Params ###########
#################################
model_dir=exp/chain_cleaned/tdnn_1d_sp
								# Kaldi chain model dir.
extra_left_context=0			# Extra contexts. Only needs if 
extra_right_context=0			#  model be (b)lstm or transformer.
frames_per_chunk=150			# 
online_ivector_dir=exp/nnet3_cleaned     		
								# not required. Kaldi online ivectors.
#################################

#################################
## MAIN experiment parameters ###
#################################

# Output:
exp_dir=exp/ltlm
								# Output experiment dir
#################################

#################################
#### Parameters for stage 0 #####
#################################
extra_texts=data/local/lm/librispeech-lm-norm.train.txt
fam_train_utts="600"  					
								# If not None fam model will training on combine of this dir 
utts_per_split=301725           # extra text will be split by N of the same size as the acoustic train 
# Output:
fam_train_dir=data/train_fam_hires
								# Directory for combining data dirs for training fake AM
dir_with_extra_train_dirs=data/extra_text
								# Will be created. Contains kaldi-like directories with splited and balanced text 
#################################

#################################
# Parameters for stage 1 decode #
#################################
acwt=1.0
post_decode_acwt=10.0
decode_num_threads=12
decode_use_gpu=true
decode_train_nj=1200
# Output:
# exp_dir/train.decoded and test.decoded files
# which contains "data_dir lats" pairs
#################################

###################################
# Parameters for stage 1 generate #
###################################
beam=13                 # Decode beam for generating lattices. Recommended 11 < beam < 15
max_active=3000         # max active hpys for generating lattices.  
generate_nj=1200		# Lattice will generated in generate_nj process

# Output:
# exp_dir/train.generated file
# which contains "data_dir generated_lats" pairs
###################################

######################################
# Parameters for stage 2 prepare egs #
######################################
lmwt=11           					# Lm scale for prunning
prune_beam=4						# Prunning lattices this prune_beam before lattice transformer
training_type='oracle_path' 		# Can be oracle_path or choices_at_fork
egs_basename=lt_egs_$training_type	# Subdirectory name
# Output:
# For each lats in exp_dir/train.decoded test.decoded and train.generated
# make subdirectory with egs
#######################################

#################################################################
########################### END PARAMS ##########################
#################################################################

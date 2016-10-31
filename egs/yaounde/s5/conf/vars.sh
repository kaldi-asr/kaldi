# BNF training parameters
bnf_num_hidden_layers=6
bottleneck_dim=42
bnf_hidden_layer_dim=2048
bnf_minibatch_size=512
bnf_init_learning_rate=0.008
bnf_final_learning_rate=0.0008
bnf_max_change=40
bnf_num_jobs=4
bnf_num_threads=1
bnf_mixup=10000
bnf_mpe_learning_rate=0.00009
bnf_mpe_last_layer_factor=0.1
bnf_num_gauss_ubm=550 # use fewer UBM Gaussians than the
                      # non-bottleneck system (which has 800)
bnf_num_gauss_sgmm=50000 # use fewer SGMM sub-states than the
                         # non-bottleneck system (which has 80000).
bnf_decode_acwt=0.066666


# DNN hybrid system training parameters
dnn_num_hidden_layers=4
dnn_input_dim=4000
dnn_output_dim=400
dnn_init_learning_rate=0.008
dnn_final_learning_rate=0.0008
dnn_mixup=12000

dnn_mpe_learning_rate=0.00008
dnn_mpe_last_layer_factor=0.1
dnn_mpe_retroactive=true

bnf_every_nth_frame=2 # take every 2nd frame.
babel_type=full

use_pitch=true

lmwt_plp_extra_opts=( --min-lmwt 8 --max-lmwt 12 )
lmwt_bnf_extra_opts=( --min-lmwt 15 --max-lmwt 22 )
lmwt_dnn_extra_opts=( --min-lmwt 10 --max-lmwt 15 )

dnn_beam=16.0
dnn_lat_beam=8.5

icu_opt=(--use-icu true --icu-transform Any-Lower)

if [[ `hostname` == *.tacc.utexas.edu ]] ; then
  decode_extra_opts=( --num-threads 4 --parallel-opts "-pe smp 4" )
  sgmm_train_extra_opts=( )
  sgmm_group_extra_opts=( --num_iters 25 ) 
  sgmm_denlats_extra_opts=( --num-threads 2 )
  sgmm_mmi_extra_opts=(--cmd "local/lonestar.py -pe smp 2")
  dnn_denlats_extra_opts=( --num-threads 2 )

  dnn_cpu_parallel_opts=(--minibatch-size 128 --num-jobs-nnet 8 --num-threads 16 \
                         --parallel-opts "-pe smp 16" )
  dnn_gpu_parallel_opts=(--minibatch-size 512 --num-jobs-nnet 8 --num-threads 1)

  dnn_gpu_mpe_parallel_opts=(--num-jobs-nnet 8 --num-threads 1)
  dnn_gpu_mpe_parallel_opts=(--num-jobs-nnet 8 --num-threads 1)
  dnn_parallel_opts="-l gpu=1"
else
  decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
  sgmm_train_extra_opts=( --num-iters 25 )
  sgmm_group_extra_opts=(--group 3 --parallel-opts "-pe smp 3 -l mem_free=7G,ram_free=2.75G" --cmd "run.pl") 
  sgmm_denlats_extra_opts=(--num-threads 4 --parallel-opts "-pe smp 4" --cmd "run.pl")
  sgmm_mmi_extra_opts=(--cmd "run.pl")
  dnn_denlats_extra_opts=(--num-threads 4 --parallel-opts "-pe smp 4" --cmd "run.pl -l")

  dnn_cpu_parallel_opts=(--minibatch-size 128 --num-jobs-nnet 8 --num-threads 16 \
                         --parallel-opts "-pe smp 16" --cmd "run.pl")
  dnn_gpu_parallel_opts=(--minibatch-size 512 --num-jobs-nnet 8 --num-threads 1 \
                         --parallel-opts "-l gpu=1" --cmd "run.pl")
  dnn_parallel_opts="-l gpu=1"
  dnn_gpu_mpe_parallel_opts=(--num-jobs-nnet 8 --num-threads 1 \
                             --parallel-opts "-l gpu=1" --cmd "run.pl")
fi
 
icu_transform="Any-Lower"
case_insensitive=true


max_states=150000
wip=0.5


phoneme_mapping=

minimize=true

proxy_phone_beam=-1
proxy_phone_nbest=-1
proxy_beam=5
proxy_nbest=500

extlex_proxy_phone_beam=5
extlex_proxy_phone_nbest=300
extlex_proxy_beam=-1
extlex_proxy_nbest=-1
#keyword search default
glmFile=conf/glm
duptime=0.5
case_insensitive=false
use_pitch=true
# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="-oov <unk>"
boost_sil=1.5 #  note from Dan: I expect 1.0 might be better (equivalent to not
              # having the option)... should test.
cer=0

#Declaring here to make the definition inside the language conf files more
# transparent and nice
declare -A dev10h_more_kwlists
declare -A dev2h_more_kwlists
declare -A eval_more_kwlists
declare -A shadow_more_kwlists

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source train and decode cmds.
nj=10
train_nj=10
# data locations
gp_train_data=/mnt/corpora/Globalphone/gp/FRF_ASR003/wav
yaounde_train_data=/mnt/corpora/Yaounde/read/wavs/16000
ca_test_data=/mnt/corpora/central_accord/test
answers_data=/mnt/corpora/Yaounde/answers/16000

# tools
tok_home=/home/tools/mosesdecoder/scripts/tokenizer
lowercaser=$tok_home/lowercase.perl
normalizer="$tok_home/normalize-punctuation.perl -l fr"
tokenizer="$tok_home/tokenizer.perl -l fr"
deescaper=$tok_home/deescape-special-chars.perl
lm_dir=../lm/data/local/lm

#speech corpora files location
train_data_dir=/home/data/babel/data/101-cantonese/release-current/conversational/training
train_data_list=/home/data/babel/data/splits/Cantonese_Babel101/train.FullLP.list

#RADICAL DEV data files
dev2h_data_dir=/home/data/babel/data/101-cantonese/release-current/conversational/dev
dev2h_data_list=/home/data/babel/data/splits/Cantonese_Babel101/dev.3hr.list
dev2h_data_cmudb=/home/data/babel/data/splits/Cantonese_Babel101/uem/db-v8-utt.dat
dev2h_stm_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev/IARPA-babel101b-v0.4c_conv-dev.stm
dev2h_ecf_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.ecf.xml
dev2h_rttm_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev/IARPA-babel101b-v0.4c_conv-dev.mitllfa2.rttm
dev2h_kwlist_file=/home/data/babel/data/splits/Cantonese_Babel101/babel101b-v0.4c_conv-dev.kwlist.xml
dev2h_more_kwlists=(
                      [dev]=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.kwlist.xml
                      [eval]=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.kwlist2.xml
)
dev2h_subset_ecf=true
dev2h_nj=20

#Official DEV data files
dev10h_data_dir=/home/data/babel/data/101-cantonese/release-current/conversational/dev
dev10h_data_list=/home/data/babel/data/splits/Cantonese_Babel101/dev.list
dev10h_data_cmudb=/home/data/babel/data/splits/Cantonese_Babel101/uem/db-v8-utt.dat
dev10h_stm_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev/IARPA-babel101b-v0.4c_conv-dev.stm
dev10h_ecf_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.ecf.xml
dev10h_rttm_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev/IARPA-babel101b-v0.4c_conv-dev.mitllfa2.rttm
dev10h_kwlist_file=/home/data/babel/data/splits/Cantonese_Babel101/babel101b-v0.4c_conv-dev.kwlist.xml
dev10h_more_kwlists=(
                      [dev]=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.kwlist.xml
                      [eval]=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.kwlist2.xml
)
dev10h_nj=32


#Official EVAL period evaluation data files
eval_data_dir=/home/data/babel/data/101-cantonese/release-current/conversational/eval
eval_data_list=/home/data/babel/data/splits/Cantonese_Babel101/eval.babel101b-v0.4c.list
eval_data_cmudb=/home/data/babel/data/splits/Cantonese_Babel101/uem/db-v8-utt.dat
eval_ecf_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-eval.ecf.xml
eval_kwlist_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-eval.kwlist.xml
eval_nj=64

#Shadow data files 
shadow_data_dir=(
                /home/data/babel/data/101-cantonese/release-current/conversational/dev
                /home/data/babel/data/101-cantonese/release-current/conversational/eval
              )
shadow_data_cmudb=/home/data/babel/data/splits/Cantonese_Babel101/uem/db-v8-dev+eval.utt.dat
shadow_data_list=(
                /home/data/babel/data/splits/Cantonese_Babel101/dev.list
                /home/data/babel/data/splits/Cantonese_Babel101/eval.babel101b-v0.4c.list
              )
shadow_ecf_file=/home/data/babel/data/scoring/IndusDB/IARPA-babel101b-v0.4c_conv-dev.ecf.xml
shadow_kwlist_file=/home/data/babel/data/splits/Cantonese_Babel101/babel101b-v0.4c_conv-dev.kwlist.xml
shadow_more_kwlists=(
                      [dev]=/home/data/babel/data/scoring/IndusDB/IARPA-babel104b-v0.4bY_conv-dev.kwlist.xml
                      [eval]=/home/data/babel/data/scoring/IndusDB/IARPA-babel104b-v0.4bY_conv-dev.kwlist2.xml

                    )
shadow_nj=64


# Acoustic model parameters
numLeavesTri1=1000
numGaussTri1=10000
numLeavesTri2=1000
numGaussTri2=20000
numLeavesTri3=6000
numGaussTri3=75000
numLeavesMLLT=6000
numGaussMLLT=75000
numLeavesSAT=6000
numGaussSAT=75000
numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=80000

# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="--romanized --oov <unk>"

# Scoring protocols (dummy GLM file to appease the scoring script)
glmFile=/home/data/babel/data/splits/Cantonese_Babel101/cantonese.glm
lexicon_file=/home/data/babel/data/101-cantonese/release-current/conversational/reference_materials/lexicon.txt
cer=1

max_index_states=150000
word_ins_penalty=0.5

#keyword search settings
duptime=0.5
case_insensitive=true

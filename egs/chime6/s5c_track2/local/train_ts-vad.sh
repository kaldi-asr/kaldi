#!/bin/bash
# Copyright  2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0

# This script trains TS-VAD model using the same training data
# as in the baseline acoustic model.

. ./path.sh
. ./cmd.sh

# Begin configuration section.
stage=0
train_stage=-10
srand=0

# Training options
num_epochs=2
lrate=0003
l2=0.002
l2o=0.001
common_egs_dir=
remove_egs=true

lang=data/lang
silphonelist=1:2:3:4:5:21:22:23:24:25
spnphonelist=

sa=60 #number of seconds to sub-split speakers
basedata=train_worn_simu_u400k_cleaned_sp
srcdata=${basedata}_${sa}s
data=${srcdata}_hires
lats=${PWD}/exp/tri3_cleaned_ali_${basedata}
nnet3_affix=_train_worn_simu_u400k_cleaned_rvb
affix=1a

tardir=$lats/VAD_targets
targets=$tardir/dense-4H/dense_targets.scp
ivector_dir=${PWD}/exp/nnet3${nnet3_affix}
nj_ivec=128
nj_paste=48
dir=exp/ts-vad_$affix

chime6_corpus=${PWD}/CHiME6
json_dir=${chime6_corpus}/transcriptions
json_ali=${PWD}/data/json_ali
sess_list="S03 S04 S05 S06 S07 S08 S12 S13 S16 S17 S18 S19 S20 S22 S23 S24"
sess_num=16

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mdl=$lats/final.mdl
[ ! -f $mdl ] && echo "$0: expected model file $mdl to exist!" && exit 1;
ivdir=$ivector_dir/ivectors-offline_${data}
iv4dir=$ivector_dir/ivectors-offline-4spk_${data}

if [ $stage -le 0 ]; then
  if [ ! -f data/${srcdata}_hires/.done ]; then
    echo "Splitting speakers in ${basedata} into ${sa}-second subspeakers"
    utils/data/modify_speaker_info.sh --seconds-per-spk-max $sa data/${basedata}_hires data/${srcdata}_hires
    touch data/${srcdata}_hires/.done
  fi
fi

if [ $stage -le 1 ]; then
  outdir=$tardir
  nj=$(cat $lats/num_jobs) || exit 1;
  if [ -f $lats/ali.1.gz ]; then
    if [ ! -f $outdir/.done ]; then
      echo "Preparing per-utterance 1-speaker VAD targets from alignment"
      $train_cmd JOB=1:$nj $outdir/log/ali_to_phones.JOB.log \
        gunzip -c $lats/ali.JOB.gz \| \
          ali-to-phones --frame-shift=0.01 --per-frame=true ${mdl} ark:- ark,t:$outdir/ali_phones.JOB.ark || exit 1;
      $train_cmd JOB=1:$nj $outdir/log/conv_ali_to_vad.JOB.log \
        python3 local/ts-vad/conv_ali_to_vad_012.py "$silphonelist" "$spnphonelist" $outdir/ali_phones.JOB.ark $outdir/ali_vad_targets.JOB.ark || exit 1
      cat $outdir/ali_vad_targets.*.ark | sort > $outdir/ali_vad_targets.ark
      if [ ! -f $outdir/targets.ark ]; then
        vali_dst="ark,scp:$outdir/targets.ark,$outdir/targets.scp"
        copy-int-vector "ark:$outdir/ali_vad_targets.ark" "$vali_dst" || exit 1
      fi
      touch $outdir/.done
    fi
  fi
fi

if [ $stage -le 2 ]; then
  if [ ! -f $json_ali/.done ]; then
    echo "Converting JSON to per-session VAD alignment (overlapped speech is considered as silence, to exclude these regions from i-vectors estimation)"
    mkdir -p $json_ali
    for json in `find $json_dir/ -name "*.json"`; do
      sess=$(basename $json | sed s:.json::)
      echo $sess
      $train_cmd $json_ali/${sess}.log \
        python local/ts-vad/make_json_align.py $json ark,t,scp:$json_ali/$sess.ark,$json_ali/${sess}.scp || exit 1;
      $train_cmd $json_ali/${sess}_sp0.9.log \
        python local/ts-vad/make_json_align.py --frame_shift 0.009 $json ark,t,scp:$json_ali/${sess}_sp0.9.ark,$json_ali/${sess}_sp0.9.scp || exit 1;
      sed -i s:\ :_sp0.9\ : $json_ali/${sess}_sp0.9.scp
      $train_cmd $json_ali/${sess}_sp1.1.log \
        python local/ts-vad/make_json_align.py --frame_shift 0.011 $json ark,t,scp:$json_ali/${sess}_sp1.1.ark,$json_ali/${sess}_sp1.1.scp || exit 1;
      sed -i s:\ :_sp1.1\ : $json_ali/${sess}_sp1.1.scp
    done
    cat $json_ali/*.scp > $json_ali/all_sess.scp
    touch $json_ali/.done
  fi
fi

if [ $stage -le 3 ]; then
  ivdata=${srcdata}_hires
  outdir=$ivdir
  if [ ! -f $outdir/.lats-weights.done ]; then
    echo 'Preparing weights for i-vectors extraction from ali/lats'
    silence_weight=0.00001
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    acwt=0.1
    if [ ! -f $lats/final.mdl ]; then
      echo "$0: expected $lats/final.mdl to exist."
      exit 1;
    fi
    if [ -f $lats/ali.1.gz ]; then
      nj_orig=$(cat $lats/num_jobs) || exit 1;
      rm $outdir/weights.*.gz 2>/dev/null
      $train_cmd JOB=1:$nj_orig  $outdir/log/ali_to_post.JOB.log \
        gunzip -c $lats/ali.JOB.gz \| \
        ali-to-post ark:- ark:- \| \
        weight-silence-post $silence_weight $silphonelist $lats/final.mdl ark:- ark:- \| \
        post-to-weights ark:- "ark,t:|gzip -c >$outdir/weights.JOB.gz" || exit 1;
      for j in $(seq $nj_orig); do gunzip -c $outdir/weights.$j.gz; done | gzip -c >$outdir/weights_lats.gz || exit 1;
      rm $outdir/weights.*.gz || exit 1;
    elif [ -f $lats/lat.1.gz ]; then
      rm $outdir/weights.*.gz 2>/dev/null
      $train_cmd JOB=1:$nj_orig  $outdir/log/lat_to_post.JOB.log \
        lattice-best-path --acoustic-scale=$acwt "ark:gunzip -c $lats/lat.JOB.gz|" ark:/dev/null ark:- \| \
        ali-to-post ark:- ark:- \| \
        weight-silence-post $silence_weight $silphonelist $lats/final.mdl ark:- ark:- \| \
        post-to-weights ark:- "ark:|gzip -c >$outdir/weights.JOB.gz" || exit 1;
      for j in $(seq $nj_orig); do gunzip -c $outdir/weights.$j.gz; done | gzip -c >$outdir/weights_lats.gz || exit 1;
      rm $outdir/weights.*.gz || exit 1;
    else
      echo "$0: expected ali.1.gz or lat.1.gz to exist in $lats";
      exit 1;
    fi
    touch $outdir/.lats-weights.done
  fi
  if [ ! -f $outdir/.json-weights.done ]; then
    echo 'Preparing weights for i-vectors extraction from json'
    perl local/ts-vad/prepare_json_weights.pl data/$ivdata/segments $json_ali/all_sess.scp $outdir/weights_json.scp || exit 1;
    touch $outdir/.json-weights.done
  fi
  if [ ! -f $outdir/.mult-weights.done ]; then
    echo 'Multiplying weights from lats and json'
    $train_cmd $outdir/multiply-vectors.log \
      multiply-vectors --length-tolerance=2 ark:"gunzip -c $outdir/weights_lats.gz |" scp:$outdir/weights_json.scp ark,t:"| gzip -c >$outdir/weights_mult.gz" || exit 1;
    touch $outdir/.mult-weights.done
  fi
  if [ ! -f $outdir/.done ]; then
    echo 'Preparing single-speaker offline i-vectors'
    local/ts-vad/extract_ivectors.sh --cmd $train_cmd --nj $nj_ivec \
    --sub-speaker-frames 0 --max-count 100 \
    data/$ivdata $lang $ivector_dir/extractor $outdir/weights_mult.gz $outdir || exit 1;
    touch $outdir/.done
  fi
fi

if [ $stage -le 4 ]; then
  outdir=$iv4dir
  if [ ! -f $outdir/.done ]; then
    mkdir -p $outdir
    echo 'Preparing 4-speaker i-vectors'
    if [ ! -f data/${srcdata}_hires/utt2spk_cl3 ]; then
      echo 'Creating 3 negative utt2spk files with speakers from the same session'
      local/ts-vad/make_negative_utt2spk.pl data/${srcdata}_hires/utt2spk \
        data/${srcdata}_hires/utt2spk_cl1 data/${srcdata}_hires/utt2spk_cl2 data/${srcdata}_hires/utt2spk_cl3 || exit 1;
    fi

    cat $ivdir/ivectors_spk.*.ark > $outdir/ivectors_spk.ark
    $train_cmd JOB=1:3 $outdir/log/apply-map.JOB.log \
      local/ts-vad/apply_map.pl --permissive -f 2 $outdir/ivectors_spk.ark \<data/${srcdata}_hires/utt2spk_clJOB \| \
      copy-vector ark:- ark,t,scp:$outdir/ivectors_utt_negJOB.ark,$outdir/ivectors_utt_negJOB.scp || exit 1;
    ivector_dim=$[$(head -n 1 $ivdir/ivectors_spk.1.ark | wc -w) - 3] || exit 1;
    base_feat_dim=$(feat-to-dim scp:data/${srcdata}_hires/feats.scp -) || exit 1;
    start_dim=$base_feat_dim
    end_dim=$[$base_feat_dim+4*$ivector_dim-1]
    absdir=$(utils/make_absolute.sh $outdir)
    cp $ivdir/{ivector_period,final.ie.id} $outdir/
    ivector_period=$(cat $ivdir/ivector_period)

    [ ! -f $outdir/ivectors_utt_pos.scp ] && copy-vector ark:"cat $ivdir/ivectors_utt.*.ark |" ark,t,scp:$outdir/ivectors_utt_pos.ark,$outdir/ivectors_utt_pos.scp

    if [ ! -f data/${srcdata}_hires/utt2spk_shuf4 ]; then
      echo 'Shuffling original and 3 negative i-vectors and utt2spk'
      local/ts-vad/shuffle_4spk_scp_utt2spk.pl $outdir/ivectors_utt_pos.scp $outdir/ivectors_utt_neg1.scp $outdir/ivectors_utt_neg2.scp $outdir/ivectors_utt_neg3.scp \
        $outdir/ivectors_utt_shuf.1.scp $outdir/ivectors_utt_shuf.2.scp $outdir/ivectors_utt_shuf.3.scp $outdir/ivectors_utt_shuf.4.scp \
        data/${srcdata}_hires/utt2spk data/${srcdata}_hires/utt2spk_cl1 data/${srcdata}_hires/utt2spk_cl2 data/${srcdata}_hires/utt2spk_cl3 \
        data/${srcdata}_hires/utt2spk_shuf1 data/${srcdata}_hires/utt2spk_shuf2 data/${srcdata}_hires/utt2spk_shuf3 data/${srcdata}_hires/utt2spk_shuf4 || exit 1;
    fi

    utils/split_data.sh data/${srcdata}_hires $nj_paste

    $train_cmd JOB=1:$nj_paste $outdir/log/paste_vectors.JOB.log \
      paste-vectors scp:"utils/filter_scp.pl data/${srcdata}_hires/split$nj_paste/JOB/utt2spk $outdir/ivectors_utt_shuf.1.scp |" \
                    scp:"utils/filter_scp.pl data/${srcdata}_hires/split$nj_paste/JOB/utt2spk $outdir/ivectors_utt_shuf.2.scp |" \
                    scp:"utils/filter_scp.pl data/${srcdata}_hires/split$nj_paste/JOB/utt2spk $outdir/ivectors_utt_shuf.3.scp |" \
                    scp:"utils/filter_scp.pl data/${srcdata}_hires/split$nj_paste/JOB/utt2spk $outdir/ivectors_utt_shuf.4.scp |" \
                    ark,scp:$outdir/ivectors_utt_4ivc.JOB.ark,$outdir/ivectors_utt_4ivc.JOB.scp || exit 1;

    $train_cmd JOB=1:$nj_paste $outdir/log/duplicate_feats.JOB.log \
      append-vector-to-feats scp:data/${srcdata}_hires/split$nj_paste/JOB/feats.scp scp:$outdir/ivectors_utt_4ivc.JOB.scp ark:- \| \
      select-feats "$start_dim-$end_dim" ark:- ark:- \| \
      subsample-feats --n=$ivector_period ark:- ark:- \| \
      copy-feats --compress=true ark:- \
      ark,scp:$absdir/ivector_online.JOB.ark,$absdir/ivector_online.JOB.scp || exit 1;

    cat $outdir/ivector_online.*.scp | sort > $outdir/ivector_online.scp
    touch $outdir/.done
  fi
fi

if [ $stage -le 5 ]; then
  outdir=$(dirname $targets)
  mkdir -p $outdir
  tmp=$(dirname $outdir)
  mkdir -p $tmp/tmp_sess
  nj=$(cat $lats/num_jobs) || exit 1;
  if [ ! -f $outdir/.done ]; then
    echo 'Creating 8-dimensional dense targets for TS-VAD training'
    [ ! -f $tmp/ali_vad_targets_wk.ark ] && grep -v "rev" $tmp/ali_vad_targets.ark > $tmp/ali_vad_targets_wk.ark
    [ ! -f $tmp/segments_wk_ali ] && utils/filter_scp.pl $tmp/ali_vad_targets_wk.ark data/${srcdata}_hires/segments > $tmp/segments_wk_ali
    [ ! -f data/${srcdata}_hires/utt2num_frames ] && feat-to-len scp:data/${srcdata}_hires/feats.scp ark,t:data/${srcdata}_hires/utt2num_frames

    for p in `seq 4`; do
      [ ! -f $tmp/utt2spk_shuf${p} ] && cat data/${srcdata}_hires/utt2spk_shuf${p} | sort > $tmp/utt2spk_shuf${p}
    done

    nj_dense=$((sess_num*3))
    j=0
    for sess in $sess_list; do
      jp=$((3*j+1))
      for p in `seq 4`; do
        [ ! -f $tmp/tmp_sess/utt2spk_shuf${p}.$jp ] && grep "$sess" $tmp/utt2spk_shuf${p} | grep -v "sp" > $tmp/tmp_sess/utt2spk_shuf${p}.$jp
      done
      [ ! -f $tmp/tmp_sess/ali_vad_targets_wk.$jp.ark ] && grep "$sess" $tardir/ali_vad_targets_wk.ark | grep -v "sp" > $tmp/tmp_sess/ali_vad_targets_wk.$jp.ark

      jp=$((3*j+2))
      for p in `seq 4`; do
        [ ! -f $tmp/tmp_sess/utt2spk_shuf${p}.$jp ] && grep "$sess" $tmp/utt2spk_shuf${p} | grep "sp0.9" > $tmp/tmp_sess/utt2spk_shuf${p}.$jp
      done
      [ ! -f $tmp/tmp_sess/ali_vad_targets_wk.$jp.ark ] && grep "$sess" $tardir/ali_vad_targets_wk.ark | grep "sp0.9" > $tmp/tmp_sess/ali_vad_targets_wk.$jp.ark

      jp=$((3*j+3))
      for p in `seq 4`; do
        [ ! -f $tmp/tmp_sess/utt2spk_shuf${p}.$jp ] && grep "$sess" $tmp/utt2spk_shuf${p} | grep "sp1.1" > $tmp/tmp_sess/utt2spk_shuf${p}.$jp
      done
      [ ! -f $tmp/tmp_sess/ali_vad_targets_wk.$jp.ark ] && grep "$sess" $tardir/ali_vad_targets_wk.ark | grep "sp1.1" > $tmp/tmp_sess/ali_vad_targets_wk.$jp.ark
      j=$((j+1))
    done

    $train_cmd JOB=1:$nj_dense $outdir/log/prepare_targets.JOB.log \
      python3 local/ts-vad/conv_vad_to_dense_targets.py $tmp/tmp_sess/ali_vad_targets_wk.JOB.ark "ark,t,scp:$outdir/dense_targets.JOB.ark,$outdir/dense_targets.JOB.scp" \
      $tmp/tmp_sess/utt2spk_shuf1.JOB $tmp/tmp_sess/utt2spk_shuf2.JOB $tmp/tmp_sess/utt2spk_shuf3.JOB $tmp/tmp_sess/utt2spk_shuf4.JOB \
      data/${srcdata}_hires/segments $tmp/segments_wk_ali data/${srcdata}_hires/utt2num_frames || exit 1;
    cat $outdir/dense_targets.*.scp | sort > $targets

    # some diagnostics
    compute-cmvn-stats scp:$outdir/dense_targets.1.scp - | cmvn-to-nnet - $outdir/S03.cmvn.nnet
    compute-cmvn-stats scp:$outdir/dense_targets.2.scp - | cmvn-to-nnet - $outdir/S03_sp0.9.cmvn.nnet
    compute-cmvn-stats scp:$outdir/dense_targets.3.scp - | cmvn-to-nnet - $outdir/S03_sp1.1.cmvn.nnet

    touch $outdir/.done
  fi
fi

if [ $stage -le 14 ]; then
  mark=$dir/.done_cfg
  if [ ! -f $mark ]; then
    echo "Creating neural net configs using the xconfig parser"
    feat_dim=40
    num_targets=8
    mkdir -p $dir/configs
    output_opts="l2-regularize=$l2o"
    lstm_opts="l2-regularize=$l2"
    linear_opts="l2-regularize=$l2 orthonormal-constraint=-1.0"
    cnn_opts="l2-regularize=$l2"

    rproj=128
    nproj=32
    cell=896
    cat <<EOF > $dir/configs/network.xconfig
    input dim=400 name=ivector
    input dim=${feat_dim} name=input
    idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
    batchnorm-component name=batchnorm input=idct

    stats-layer name=mean config=mean(-150:1:1:150) input=batchnorm
    no-op-component name=batchnorm-cmn input=Sum(batchnorm,Scale(-1.0,mean))

    no-op-component name=ivector-all input=ReplaceIndex(ivector,t,0)
    dim-range-component name=ivector-1 input=ivector-all dim=100 dim-offset=0
    dim-range-component name=ivector-2 input=ivector-all dim=100 dim-offset=100
    dim-range-component name=ivector-3 input=ivector-all dim=100 dim-offset=200
    dim-range-component name=ivector-4 input=ivector-all dim=100 dim-offset=300

    combine-feature-maps-layer name=combine_inputs input=Append(batchnorm, batchnorm-cmn) num-filters1=1 num-filters2=1 height=$feat_dim
    conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
    conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
    conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
    conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128

    linear-component $linear_opts name=aff1 input=Append(cnn4,ivector-1) dim=$((3*rproj))
    fast-lstmp-layer name=blstm1-1-forward input=aff1 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-1 $lstm_opts
    fast-lstmp-layer name=blstm1-1-backward input=aff1 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=1 $lstm_opts
    fast-lstmp-layer name=blstm2-1-forward input=Append(blstm1-1-forward, blstm1-1-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-2 $lstm_opts
    fast-lstmp-layer name=blstm2-1-backward input=Append(blstm1-1-forward, blstm1-1-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=2 $lstm_opts

    linear-component $linear_opts name=aff2 input=Append(cnn4,ivector-2) dim=$((3*rproj))
    fast-lstmp-layer name=blstm1-2-forward input=aff2 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-1 $lstm_opts
    fast-lstmp-layer name=blstm1-2-backward input=aff2 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=1 $lstm_opts
    fast-lstmp-layer name=blstm2-2-forward input=Append(blstm1-2-forward, blstm1-2-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-2 $lstm_opts
    fast-lstmp-layer name=blstm2-2-backward input=Append(blstm1-2-forward, blstm1-2-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=2 $lstm_opts

    linear-component $linear_opts name=aff3 input=Append(cnn4,ivector-3) dim=$((3*rproj))
    fast-lstmp-layer name=blstm1-3-forward input=aff3 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-1 $lstm_opts
    fast-lstmp-layer name=blstm1-3-backward input=aff3 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=1 $lstm_opts
    fast-lstmp-layer name=blstm2-3-forward input=Append(blstm1-3-forward, blstm1-3-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-2 $lstm_opts
    fast-lstmp-layer name=blstm2-3-backward input=Append(blstm1-3-forward, blstm1-3-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=2 $lstm_opts

    linear-component $linear_opts name=aff4 input=Append(cnn4,ivector-4) dim=$((3*rproj))
    fast-lstmp-layer name=blstm1-4-forward input=aff4 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-1 $lstm_opts
    fast-lstmp-layer name=blstm1-4-backward input=aff4 cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=1 $lstm_opts
    fast-lstmp-layer name=blstm2-4-forward input=Append(blstm1-4-forward, blstm1-4-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-2 $lstm_opts
    fast-lstmp-layer name=blstm2-4-backward input=Append(blstm1-4-forward, blstm1-4-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=2 $lstm_opts

    fast-lstmp-layer name=blstm3-forward input=Append(blstm2-1-forward, blstm2-1-backward, blstm2-2-forward, blstm2-2-backward, blstm2-3-forward, blstm2-3-backward, blstm2-4-forward, blstm2-4-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=-3 $lstm_opts
    fast-lstmp-layer name=blstm3-backward input=Append(blstm2-1-forward, blstm2-1-backward, blstm2-2-forward, blstm2-2-backward, blstm2-3-forward, blstm2-3-backward, blstm2-4-forward, blstm2-4-backward) cell-dim=$cell recurrent-projection-dim=$rproj non-recurrent-projection-dim=$nproj delay=3 $lstm_opts

    output-layer $output_opts input=Append(blstm3-forward, blstm3-backward) name=output dim=2
    output-layer $output_opts input=Append(blstm3-forward, blstm3-backward) name=output2 dim=2
    output-layer $output_opts input=Append(blstm3-forward, blstm3-backward) name=output3 dim=2
    output-layer $output_opts input=Append(blstm3-forward, blstm3-backward) name=output4 dim=2
EOF
    steps/nnet3/xconfig_to_configs.py \
      --xconfig-file $dir/configs/network.xconfig \
      --config-dir $dir/configs || exit 1
    echo "num_targets=$num_targets" >> $dir/configs/vars

    echo "Modifying final.config file to combine 4 softmax layers in the output-node "
    sed -i 's:output\.:output1\.:g' $dir/configs/final.config
    mv $dir/configs/final.config $dir/configs/final.config.tmp
    grep -v "output\-node" $dir/configs/final.config.tmp > $dir/configs/final.config
    echo "output-node name=output input=Append(output1.log-softmax, output2.log-softmax, output3.log-softmax, output4.log-softmax)" >> $dir/configs/final.config

    echo "Modifying final.config file to enforce weigths sharing in affine and blstm layers"
    sed -i s:component\ name=aff1:component\ name=aff-uni: $dir/configs/final.config
    sed -i s:component=aff1:component=aff-uni: $dir/configs/final.config
    sed -i s:component=aff2:component=aff-uni: $dir/configs/final.config
    sed -i s:component=aff3:component=aff-uni: $dir/configs/final.config
    sed -i s:component=aff4:component=aff-uni: $dir/configs/final.config
    sed -i s:component\ name=blstm1-1:component\ name=blstm1-uni: $dir/configs/final.config
    sed -i s:component\ name=blstm2-1:component\ name=blstm2-uni: $dir/configs/final.config
    sed -i s:component=blstm1-1:component=blstm1-uni: $dir/configs/final.config
    sed -i s:component=blstm1-2:component=blstm1-uni: $dir/configs/final.config
    sed -i s:component=blstm1-3:component=blstm1-uni: $dir/configs/final.config
    sed -i s:component=blstm1-4:component=blstm1-uni: $dir/configs/final.config
    sed -i s:component=blstm2-1:component=blstm2-uni: $dir/configs/final.config
    sed -i s:component=blstm2-2:component=blstm2-uni: $dir/configs/final.config
    sed -i s:component=blstm2-3:component=blstm2-uni: $dir/configs/final.config
    sed -i s:component=blstm2-4:component=blstm2-uni: $dir/configs/final.config
    mv $dir/configs/final.config $dir/configs/final.config.tmp
    grep -v "component\ name=aff2" $dir/configs/final.config.tmp | grep -v "component\ name=aff3" |  grep -v "component\ name=aff4" | \
    grep -v "component\ name=blstm1-2" | grep -v "component\ name=blstm1-3" | grep -v "component\ name=blstm1-4" | \
    grep -v "component\ name=blstm2-2" | grep -v "component\ name=blstm2-3" | grep -v "component\ name=blstm2-4" > $dir/configs/final.config
    nnet3-init --binary=false $dir/configs/final.config $dir/configs/init.raw || exit 1;
    touch $mark
  fi
fi

if [ ! -f data/$data/utt2uniq.done ]; then
  [ -f data/$data/utt2uniq ] && mv data/$data/utt2uniq data/$data/utt2uniq.bak
  local/ts-vad/make_utt2uniq.pl data/$data/utt2spk data/$data/utt2uniq || exit 1;
  touch data/$data/utt2uniq.done
fi

if [ $stage -le 15 ]; then
  mark=$dir/.done_dnn
  if [ ! -f $mark ]; then
    cp "$(readlink -f $0)" "$dir"
    steps/nnet3/train_raw_rnn.py \
      --stage=$train_stage \
      --cmd="$train_cmd" \
      --feat.online-ivector-dir=$iv4dir \
      --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
      --trainer.srand=$srand \
      --trainer.max-param-change=2 \
      --trainer.num-epochs=$num_epochs \
      --trainer.optimization.proportional-shrink 10 \
      --trainer.optimization.momentum=0.5 \
      --trainer.optimization.num-jobs-initial=2 \
      --trainer.optimization.num-jobs-final=4 \
      --trainer.optimization.initial-effective-lrate=0.$lrate \
      --trainer.optimization.final-effective-lrate=0.0$lrate \
      --trainer.rnn.num-chunk-per-minibatch=128 \
      --trainer.samples-per-iter=15000 \
      --egs.chunk-left-context=30 \
      --egs.chunk-right-context=30 \
      --egs.chunk-width=40 \
      --use-dense-targets true \
      --feat-dir data/$data \
      --targets-scp $targets \
      --egs.cmd=run.pl \
      --egs.dir=$common_egs_dir \
      --cleanup.remove-egs false \
      --cleanup.preserve-model-interval=100 \
      --use-gpu=true \
      --dir=$dir || exit 1
    touch $mark
  fi
fi

echo Done

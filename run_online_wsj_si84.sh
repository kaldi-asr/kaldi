#!/bin/bash

. path.sh
. cmd.sh

# . ilocal/function.d/feature.sh
# . ilocal/function.d/decode.sh
# . ilocal/function.d/train.sh


datadir=data; localdir=$datadir/local; langdir=$datadir/lang;

wav_scp=/data4/wsj_dan2/s5/data/train/wav.scp

JobNum() {
  func=$FUNCNAME;
  echo -e "\n## $func $@\n"
  if [ $# -ne 2 ]; then
    echo -e "\n## $func data upperbound\n"; exit 1
  fi
  data=$1; max=$2; func=$FUNCNAME
  if [ ! -f $data/spk2utt ]; then
    echo -e "\n## $func spk2utt file is not ready in $data\n" 1>&2; exit 1
  fi
  if [ $max -le 0 ]; then
    echo -e "\n## $func max($max) is illegal\n" 1>&2 ; exit 1
  fi
  nj=`wc -l < $data/spk2utt`; 
  echo -e "\nmaximum_nj=$nj\n" 1>&2 ;
  if [ $nj -gt $max ]; then nj=$max; fi
  echo -n $nj;
}


make_wave_data(){
  wav_scp=$1; dir=$2; segments=$3;
  for f in $(cat $wav_scp|awk '{print $(NF-1);}'); do
    new_f=$( echo $f | sed "s#/data5/wsj#$dir#g"|perl -pe 's/wv[12]/wav/g;')
    path=`dirname $new_f`; [ -d $path ] || mkdir -p $path
    if [ ! -f $new_f ]; then
      sph2pipe -f wav  $f > $new_f
    fi
    shntool len $new_f | grep -v length | awk '{print $1;}' | uniq | \
    perl -e ' $f = shift @ARGV; $f =~ s/.*\///g; $f =~ s/\.wav//g;  
    $times = <STDIN>; if ($times =~ m/(\S+):(\S+)/) { $len = $2; $nframe =int( (100*$len - 2.5 )+1);  $len = ($nframe -1)*0.01; print "$f $f 0 $len\n";    } 
   '  $new_f  
  done  > $segments
}
make_wav_scp() {
  func=$FUNCNAME;
  echo -e "\n## $func $@\n"
  s_wavscp=$1; wavlist=$2;  wavscp=$3;
  cut -d " " -f1 $s_wavscp | \
  perl -e '$fname = shift @ARGV; open F, "$fname" or die "file $fname cannot open\n"; 
    while(<F>) { chomp; $lab = $_; $lab =~ s/.*\///g; $lab =~ s/\.wav//g; $vocab{$lab} = $_; }
    close F;
    while (<STDIN>) { chomp; if (exists $vocab{$_}) { print "$_  $vocab{$_}\n";  }   }
  '  $wavlist   > $wavscp || exit 1
}
make_kaldi_data() {
  func=$FUNCNAME;
  echo -e "\n## $func $@\n"
  if [ $# -ne 4 ]; then
    echo -e "\n## Example $0 sdata segments wavdir data\n"; exit 1
  fi
  sdata=$1; segments=$2; wavdir=$3 ;  data=$4; 
  [ -d $data ] || mkdir -p $data
  wlist=$wavdir/overall.wlist
  if [ ! -f $wlist ]; then
    find $wavdir  -name "*.wav" > $wlist || exit 1
  fi
  make_wav_scp $sdata/wav.scp $wlist  $data/wav.scp || exit 1
  if [ ! -f $segments ]; then echo -e "\n## $func: no file $segments \n" && exit 1; fi
  cat  $segments | perl -ne 'm/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)/; if($4 > 0){print $_; }' > $data/segments
  cp $sdata/{text,utt2spk,spk2utt} $data/
  fix_data_dir.sh $data || exit 1
}
make_lang() {
  func=$FUNCNAME;
  echo -e "\n## $func $@\n"
  if [ $# -ne 2 ];then
    echo -e "\n## Example: $0 source_lexicon  datadir \n"
  fi
  slex=$1; datadir=$2;
  langdir=$datadir/lang; localdir=$datadir/local;
  mkdir -p $datadir/{local,lang}
  cut -d " " -f2- $slex | perl -pe 's/ /\n/g;'| sort -u | \
  perl -ne 'if(/^$/||/SIL/||/NSN/||/SPN/){} else{print;} ' | \
  perl -e 'while(<STDIN>){chomp; if(m/(.*)(\d)$/){ $vocab{$1} .= " $_";   }else {print "$_\n"; }  } 
  foreach $key (keys %vocab) {
    print $key, $vocab{$key}, "\n";
  } 
  ' | perl -pe 's/^ //;s/ $//;' | \
  perl -e 'while (<STDIN>) { chomp; if(m/(\S+)\s+(\S+)/) {$vocab{$1} = $_; } 
    else { $vocab{$_} = $_ ; }}
    foreach $key (keys %vocab) { print "$vocab{$key}\n"; }
  ' | \
   sort -u  > $localdir/nonsilence_phones.txt
  (echo "SIL"; echo "<nsn>"; echo "<spn>" ) > $localdir/silence_phones.txt
  echo "SIL" > $localdir/optional_silence.txt
  cat <(echo -e "<silence>\tSIL"; echo -e "<spoken_noise>\t<spn>"; echo -e "<noise>\t<nsn>"; echo -e "<unk>\t<spn>" ) \
      <(grep -v -E ' SIL| SPN| NSN' $slex | perl -pe 'm/(\S+)\s+(.*)/; $_="$1\t$2\n";') > $localdir/lexicon.txt
  cat <(cat $localdir/silence_phones.txt| perl -ne 'chomp; print "$_ "'| perl -ne 'print "$_\n";') \
     <(cat $localdir/nonsilence_phones.txt | \
       perl -e 'while(<STDIN>) { chmop; @A = split(/\s/); 
         for($i=0; $i<@A; $i++) { $w=$A[$i]; if($w =~ m/(.*)(\d)/){ $vocab{$2} .= "$w "; } else { if (not exists $vocab1{$w}){  print "$w "; $vocab1{$w} ++;  } }  }  } 
         print "\n";  foreach $key (keys %vocab) { print "$vocab{$key}\n";  } ') > $localdir/extra_questions.txt || exit 1
  utils/prepare_lang.sh $localdir  "<spoken_noise>" $localdir/lang_tmp $langdir || exit 1;
}
make_grammar_fst() {
  func=$FUNCNAME;
  echo -e "\n## $func $@\n";
  lm=$1; lang=$2; 
  lm_suffix=`basename $lm| sed 's#\.arpa\.gz##g'`
  gzip -cd $lm | \
  utils/find_arpa_oovs.pl $lang/words.txt  > $lang/oovs_${lm_suffix}.txt

  gzip -cd $lm | \
  grep -v '<s> <s>' | \
  grep -v '</s> <s>' | \
  grep -v '</s> </s>' | \
  arpa2fst - | fstprint | \
  utils/remove_oovs.pl $lang/oovs_${lm_suffix}.txt | \
  utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang/words.txt \
  --osymbols=$lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
   fstrmepsilon > $lang/G.fst  || exit 1
  fstisstochastic $lang/G.fst 
}

sweeping_train() {
  func=$FUNCNAME;
  echo -e "\n## $func $@\n";
  norm_vars=false
  cmd=run.pl
  id=a
  totgauss=2000
  states=2000;
  pdfs=10000;
  boost_sil=1.25;
  . parse_options.sh 
  if [ $# -ne 3 ]; then
    echo -e "\n##Example $func data lang exp\n"; exit 1
  fi
  data=$1; shift
  lang=$1; shift
  exp=$1; shift

  nj=`JobNum $data 30`;  dataname=$(basename $data)
  if [ ! -f $exp/.${id}_done ]; then
    dir=$exp/mono0$id
    if [ ! -f $dir/final.mdl ]; then
      steps/train_mono.sh  --norm-vars $norm_vars --totgauss $totgauss \
      --boost-silence $boost_sil --nj $nj  --cmd "$cmd" \
      $data  $lang $dir || exit 1
    fi
    ali=$dir/ali_$dataname;
    if [ ! -f $ali/.ali_done ]; then
      steps/align_si.sh   --nj $nj --cmd "$cmd" $data  $lang $dir  $ali || exit 1
      touch $ali/.ali_done
    fi
    dir=$exp/tri1$id
    if [ ! -f $dir/final.mdl ]; then
      steps/train_deltas.sh  --norm-vars $norm_vars  --cmd "$cmd" \
      $states $pdfs  $data  $lang $ali $dir || exit 1
    fi 
    ali=$dir/ali_$dataname;
    if [ ! -f $ali/.ali_done ]; then
      steps/align_si.sh   --nj $nj --cmd "$cmd" $data  $lang $dir  $ali || exit 1
      touch $ali/.ali_done
    fi
    dir=$exp/tri2$id;
    if [ ! -f $dir/final.mdl ]; then
      steps/train_lda_mllt.sh --norm-vars $norm_vars --cmd "$cmd" \
      $states $pdfs $data  $lang $ali $dir || exit 1
    fi
    ali=$dir/ali_$dataname;
    if [ ! -f $ali/.ali_done ]; then
      steps/align_si.sh   --nj $nj --cmd "$cmd" $data  $lang $dir  $ali || exit 1
      touch $ali/.ali_done
    fi
    touch $exp/.${id}_done
  fi
}


if [ ! -f $datadir/raw/.train_done ]; then
  make_wave_data $wav_scp  $datadir/raw  $datadir/raw/train_segments  || exit 1
  touch $datadir/raw/.train_done
fi

wav_scp=/data4/wsj_dan2/s5/data/train_si84/wav.scp
if [ ! -f $datadir/raw/.train_si84_done ]; then
  make_wave_data $wav_scp  $datadir/raw  $datadir/raw/train_si84_segments  || exit 1
  touch $datadir/raw/.train_si84_done
fi

wav_scp=/data4/wsj_dan2/s5/data/test_eval92/wav.scp
if [ ! -f $datadir/raw/.eval92_done ]; then
  make_wave_data $wav_scp  $datadir/raw  $datadir/raw/eval92_segments  || exit 1
  touch $datadir/raw/.eval92_done
fi
wav_scp=/data4/wsj_dan2/s5/data/test_dev93/wav.scp
if [ ! -f $datadir/raw/.dev93_done ]; then
  make_wave_data $wav_scp  $datadir/raw  $datadir/raw/dev93_segments  || exit 1
  touch $datadir/raw/.dev93_done
fi

wavdir=data/raw
sdata=/data4/wsj_dan2/s5/data/test_dev93; 
segments=data/raw/dev93_segments
data=data/dev93;
if [ ! -f $data/.prepare_done ]; then
 make_kaldi_data $sdata $segments $wavdir $data || exit 1
 touch $data/.prepare_done
fi
sdata=/data4/wsj_dan2/s5/data/test_eval92;
segments=data/raw/eval92_segments
data=data/eval92
if [ ! -f $data/.prepare_done ]; then
  make_kaldi_data $sdata $segments $wavdir $data || exit 1
  touch $data/.prepare_done
fi
sdata=/data4/wsj_dan2/s5/data/train_si84;
segments=data/raw/train_si84_segments
data=data/train_si84

if [ ! -f $data/.prepare_done ]; then
  make_kaldi_data $sdata $segments $wavdir $data || exit 1
  touch $data/.prepare_done
fi
lex=/data4/wsj_dan2/s5/data/local/dict/lexicon.txt

if [ ! -f $langdir/.prepare_done ]; then
  make_lang  $lex    $datadir  || exit 1
  touch $langdir/.prepare_done
fi
lm=/data4/wsj_dan2/s5/data/local/nist_lm/lm_tgpr.arpa.gz
if [ ! -f $langdir/.prepare_g_done ]; then
  make_grammar_fst $lm $langdir 
  touch $langdir/.prepare_g_done  
fi
expdir=exp
if [ ! -f $datadir/mfcc/.mfcc_done ]; then
for x in train_si84 eval92 dev93; do
  sdata=$datadir/$x; data=$datadir/mfcc/$x; feat=$expdir/feature/mfcc/$x;
  [ -d $data ] || mkdir -p $data
  cp $sdata/* $data
  nj=`JobNum $sdata 30`
  steps/make_mfcc.sh --cmd "run.pl" --nj $nj  $data  $feat/_log  $feat/_data || exit 1;
  steps/compute_cmvn_stats.sh $data $feat/_log  $feat/_data || exit 1;
done
  touch $datadir/mfcc/.mfcc_done
fi
echo -e "\n## training \n";
data=$datadir/mfcc/train_si84
sweeping_train  $data $langdir $expdir || exit 1

sdir=$expdir/tri2a; graph=$sdir/graph;
if [ ! -f $graph/HCLG.fst ]; then
  utils/mkgraph.sh  $langdir $sdir $graph || exit 1
fi
for x  in eval92 dev93; do
  data=$datadir/mfcc/$x;  nj=`JobNum $data 30`;
  dir=$sdir/decode_$(basename $data)
  if [ ! -f $dir/.decode_done ]; then
    steps/decode.sh  --nj $nj $graph $data  $dir  || exit 1
   touch $dir/.decode_done
  fi
done

num_threads=1
minibatch_size=512
dir=exp/nnet2_online/nnet_gpu
data=$datadir/mfcc/train_si84;  exp1=exp/nnet2_online; train_cmd=run.pl
if [ ! -f $exp1/diag_ubm/.done ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "run.pl" --nj 10 --num-frames 200000 \
  $data 512 exp/tri2a  exp/nnet2_online/diag_ubm || exit 1
  touch $exp1/diag_ubm/.done
fi

if [ ! -f $exp1/extractor/.done ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "run.pl" --nj 4 \
    $data $exp1/diag_ubm $exp1/extractor || exit 1
   touch $exp1/extractor/.done
fi

if [ ! -f $exp1/ivectors/.done ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
  $data  exp/nnet2_online/extractor exp/nnet2_online/ivectors
  touch $exp1/ivectors/.done
fi

dir=exp/nnet2_online/nnet; sdir=$dir;
train_stage=-10
if [ ! -f $dir/.train_done ]; then
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
  --splice-width 4 \
  --feat-type raw \
  --online-ivector-dir exp/nnet2_online/ivectors \
  --cmvn-opts "--norm-means=false --norm-vars=false" \
  --num-threads "$num_threads" \
  --minibatch-size "$minibatch_size" \
  --parallel-opts "$parallel_opts" \
  --num-jobs-nnet 4 \
  --num-epochs-extra 10 --add-layers-period 1 \
  --num-hidden-layers 2 \
  --mix-up 4000 \
  --initial-learning-rate 0.02 --final-learning-rate 0.004 \
  --cmd "$train_cmd" \
  --pnorm-input-dim 1000 \
  --pnorm-output-dim 200 \
  $data  $langdir exp/tri2a/ali_train_si84 $dir  || exit 1
  touch $dir/.train_done
fi

for x in eval92 dev93; do
  data=$datadir/mfcc/$x; nj=`JobNum $data  30`; 
  dir=exp/nnet2_online/ivectors_$x;
  if [ ! -f $dir/.done ]; then
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    $data  exp/nnet2_online/extractor exp/nnet2_online/ivectors_$x || exit 1;
    touch $dir/.done
  fi
done

for x in eval92 dev93; do
  data=$datadir/mfcc/$x; nj=`JobNum $data 30`; 
  dir=$sdir/decode_$x;
  if [ ! -f $dir/.decode_done ]; then
    steps/nnet2/decode.sh --config conf/decode.config --cmd "$train_cmd" --nj $nj \
    --online-ivector-dir exp/nnet2_online/ivectors_$x \
    $graph $data $dir || exit 1  
    touch $dir/.decode_done
  fi
done

data=$datadir/mfcc/train_si84;
dir=exp/nnet2_online/nnet0b; sdir=$dir;
train_stage=-10
if [ ! -f $dir/.train_done ]; then
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
  --splice-width 4 \
  --feat-type raw \
  --cmvn-opts "--norm-means=false --norm-vars=false" \
  --num-threads "$num_threads" \
  --minibatch-size "$minibatch_size" \
  --parallel-opts "$parallel_opts" \
  --num-jobs-nnet 4 \
  --num-epochs-extra 10 --add-layers-period 1 \
  --num-hidden-layers 2 \
  --mix-up 4000 \
  --initial-learning-rate 0.02 --final-learning-rate 0.004 \
  --cmd "$train_cmd" \
  --pnorm-input-dim 1000 \
  --pnorm-output-dim 200 \
  $data  $langdir exp/tri2a/ali_train_si84 $dir  || exit 1
  touch $dir/.train_done
fi

for x in eval92 dev93; do
  data=$datadir/mfcc/$x; nj=`JobNum $data 30`; 
  dir=$sdir/decode_$x;
  if [ ! -f $dir/.decode_done ]; then
    steps/nnet2/decode.sh --config conf/decode.config --cmd "$train_cmd" --nj $nj \
    $graph $data $dir || exit 1  
    touch $dir/.decode_done
  fi
done


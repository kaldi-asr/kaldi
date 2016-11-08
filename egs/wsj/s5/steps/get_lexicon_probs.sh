#!/bin/bash
# Copyright 2013  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0


# From a training or alignment directory, and an original lexicon.txt and lang/
# directory, obtain a new lexicon with pronunciation probabilities.
# Note: this script is currently deprecated, the recipes are using a different
# script in utils/dict_dir_add_pronprobs.sh.


# Begin configuration section.  
stage=0
smooth_count=1.0 # Amount of count to add corresponding to each original lexicon entry;
                 # this corresponds to add-one smoothing of the pron-probs.
max_one=true   # If true, normalize the pron-probs so the maximum value for each word is 1.0,
               # rather than summing to one.  This is quite standard.

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: steps/get_lexicon_probs.sh <data-dir> <lang-dir> <src-dir|ali-dir> <old-lexicon> <exp-dir> <new-lexicon>"
   echo "e.g.: steps/get_lexicon_probs.sh data/train data/lang exp/tri5 data/local/lexicon.txt \\"
   echo "                      exp/tri5_lexprobs data/local_withprob/lexicon.txt"
   echo "Note: we assume you ran using word-position-dependent phones but both the old and new lexicon will not have"
   echo "these markings.  We also assume the new lexicon will have pron-probs but the old one does not; this limitation"
   echo "of the script can be removed later."
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # used to control partial re-running."
   echo "  --max-one <true|false>                           # If true, normalize so max prob of each"
   echo "                                                   # word is one.  Default: true"
   echo "  --smooth <smooth-count>                          # Amount to smooth each count by (default: 1.0)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
old_lexicon=$4
dir=$5
new_lexicon=$6

oov=`cat $lang/oov.int` || exit 1;
nj=`cat $srcdir/num_jobs` || exit 1;

for f in $data/text $lang/L.fst $lang/phones/word_boundary.int $srcdir/ali.1.gz $old_lexicon; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log
utils/split_data.sh $data $nj # Make sure split data-dir exists.
sdata=$data/split$nj

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

if [ $stage -le 0 ]; then

  ( ( for n in `seq $nj`; do gunzip -c $srcdir/ali.$n.gz; done ) | \
    linear-to-nbest ark:- "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $data/text |" '' '' ark:- | \
    lattice-align-words $lang/phones/word_boundary.int $srcdir/final.mdl ark:- ark:- | \
    lattice-to-phone-lattice --replace-words=false $srcdir/final.mdl ark:- ark,t:- | \
    awk '{ if (NF == 4) { word_phones = sprintf("%s %s", $3, $4); count[word_phones]++; } } 
        END { for(key in count) { print count[key], key; } }' | \
          sed s:0,0,:: | awk '{print $2, $1, $3;}' | sed 's/_/ /g' | \
          utils/int2sym.pl -f 3- $lang/phones.txt  | \
          sed -E 's/_I( |$)/ /g' |  sed -E 's/_E( |$)/ /g' | sed -E 's/_B( |$)/ /g' | sed -E 's/_S( |$)/ /g' | \
          utils/int2sym.pl -f 1 $lang/words.txt > $dir/lexicon_counts.txt
  ) 2>&1 | tee $dir/log/get_fsts.log

fi

cat $old_lexicon | awk '{if (!($2 > 0.0 && $2 < 1.0)) { exit(1); }}' && \
  echo "Error: old lexicon $old_lexicon appears to have pron-probs; we don't expect this." && \
  exit 1;

mkdir -p `dirname $new_lexicon` || exit 1;

if [ $stage -le 1 ]; then
  grep -v -w '^<eps>' $dir/lexicon_counts.txt | \
  perl -e ' ($old_lexicon, $smooth_count, $max_one) = @ARGV;
    ($smooth_count >= 0) || die "Invalid smooth_count $smooth_count";
    ($max_one eq "true" || $max_one eq "false") || die "Invalid max_one variable $max_one";
    open(O, "<$old_lexicon")||die "Opening old-lexicon file $old_lexicon"; 
    while(<O>) {
      $_ =~ m/(\S+)\s+(.+)/ || die "Bad old-lexicon line $_";
      $word = $1;
      $orig_pron = $2;
      # Remember the mapping from canonical prons to original prons: in the case of
      # syllable based systems we want to remember the locations of tabs in
      # the original lexicon.
      $pron = join(" ", split(" ", $orig_pron));
      $orig_pron{$word,$pron} = $orig_pron;
      $count{$word,$pron} += $smooth_count;
      $tot_count{$word} += $smooth_count;
    }
    while (<STDIN>) {
      $_ =~ m/(\S+)\s+(\S+)\s+(.+)/ || die "Bad new-lexicon line $_";
      $word = $1;
      $this_count = $2;
      $pron = join(" ", split(" ", $3));
      $count{$word,$pron} += $this_count;
      $tot_count{$word} += $this_count;
    }
    if ($max_one eq "true") {  # replace $tot_count{$word} with max count
       # of any pron.
      %tot_count = {}; # set to empty assoc array.
      foreach $key (keys %count) {
        ($word, $pron) = split($; , $key); # $; is separator for strings that index assoc. arrays.
        $this_count = $count{$key};
        if (!defined $tot_count{$word} || $this_count > $tot_count{$word}) {
          $tot_count{$word} = $this_count;
        }
      }
    }
    foreach $key (keys %count) {
       ($word, $pron) = split($; , $key); # $; is separator for strings that index assoc. arrays.
       $this_orig_pron = $orig_pron{$key};
       if (!defined $this_orig_pron) { die "Word $word and pron $pron did not appear in original lexicon."; }
       if (!defined $tot_count{$word}) { die "Tot-count not defined for word $word."; }
       $prob = $count{$key} / $tot_count{$word};
       print "$word\t$prob\t$this_orig_pron\n";  # Output happens here.
    } '  $old_lexicon $smooth_count $max_one > $new_lexicon || exit 1;
fi

exit 0;

echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;
cp $srcdir/final.occs $dir;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.


if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

## Set up model and alignment model.
mdl=$srcdir/final.mdl
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl
else
  alimdl=$srcdir/final.mdl
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;
alimdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $alimdl - |"
mdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $mdl - |"


## Work out where we're getting the graphs from.
if $use_graphs; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-graphs true, but #jobs mismatch." && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "No graphs in $srcdir" && exit 1;
  graphdir=$srcdir
else
  graphdir=$dir
  if [ $stage -le 0 ]; then
    echo "$0: compiling training graphs"
    tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";   
    $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
      compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
        "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
  fi
fi


if [ $stage -le 1 ]; then
  echo "$0: aligning data in $data using $alimdl and speaker-independent features."
  $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$alimdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$sifeats" "ark:|gzip -c >$dir/pre_ali.JOB.gz" || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing fMLLR transforms"
  if [ "$alimdl" != "$mdl" ]; then
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-post-to-gpost $alimdl "$sifeats" ark:- ark:- \| \
      gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
      ark,s,cs:- ark:$dir/trans.JOB || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
      ark,s,cs:- ark:$dir/trans.JOB || exit 1;
  fi
fi

feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"

if [ $stage -le 3 ]; then
  echo "$0: doing final alignment."
  $cmd JOB=1:$nj $dir/log/align_pass2.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

rm $dir/pre_ali.*.gz

echo "$0: done aligning data."

utils/summarize_warnings.pl $dir/log

exit 0;

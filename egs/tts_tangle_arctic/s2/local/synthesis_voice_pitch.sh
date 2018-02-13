#!/bin/sh

extra_feats=
input_xml=
input_text=
if [ "$1" == "--extra_feats" ]; then
    extra_feats=$2
    shift 2;
fi
if [ "$1" == "--input_xml" ]; then
    input_xml=$2
    shift 2;
fi
if [ "$1" == "--input_text" ]; then
    input_text=$2
    shift 2;
fi

if [ $# != 2 ]; then
  echo "Usage: synthesis_voice.sh <voicedir> <outputdir>"
  echo "This script will perform TTS using the text from STDIN using the DNN voice passed in the first argument. The audio files wil be stored in outputdir."
  exit 1
fi

synth=excitation
#synth=cere
voice_dir=$1
outdir=$2

source $voice_dir/voice.conf

cex_freq=$voice_dir/lang/cex.ark.freq
var_cmp=$voice_dir/lang/var_cmp.txt
durdnndir=$voice_dir/dur
f0dnndir=$voice_dir/pitch
dnndir=$voice_dir/acoustic
datadir=`mktemp -d`
tpdb=`readlink -f $voice_dir/lang/$tpdbvar`

[ -f path.sh ] && . ./path.sh; 

if [ ! -z "$input_xml" ]; then
    cp $input_xml $datadir/text_full.xml
else
    if [ ! -z "$input_text" ]; then
        cp $input_text $datadir/text.xml
    else
        awk 'BEGIN{print "<parent>"}{print}END{print "</parent>"}' > $datadir/text.xml
    fi

    # Generate CEX features for test set.
    idlaktxp --pretty --tpdb=$tpdb $datadir/text.xml - \
        | idlakcex --pretty --cex-arch=default --tpdb=$tpdb - $datadir/text_full.xml
fi
python local/idlak_make_lang.py --mode 2 -r "test" \
    $datadir/text_full.xml $cex_freq $datadir/cex.ark > $datadir/cex_output_dump
# Generate input feature for duration modelling
cat $datadir/cex.ark \
    | awk -v extras="$extra_feats" '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) for (state = 0; state < 5; state++) print extras, a[i], state; print "]"}' \
    | copy-feats ark:- ark,scp:$datadir/in_durfeats.ark,$datadir/in_durfeats.scp

# Duration based test set
lbldurdir=$datadir/lbldur
mkdir -p $lbldurdir
cp $datadir/in_durfeats.scp $lbldurdir/feats.scp
cut -d ' ' -f 1 $lbldurdir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $lbldurdir/utt2spk
utils/utt2spk_to_spk2utt.pl $lbldurdir/utt2spk > $lbldurdir/spk2utt
#steps/compute_cmvn_stats.sh $lbldurdir $lbldurdir $lbldurdir

# Generate label with DNN-generated duration

#  1. forward pass through duration DNN
duroutdir=`mktemp -d`
rm -rf $duroutdir
local/make_forward_fmllr.sh $durdnndir $lbldurdir $duroutdir ""
#  2. make the duration consistent, generate labels with duration information added
(echo '#!MLF!#'; for cmp in $duroutdir/cmp/*.cmp; do
    cat $cmp | awk -v nstate=5 -v id=`basename $cmp .cmp` 'BEGIN{print "\"" id ".lab\""; tstart = 0 }
{
  pd += $2;
  sd[NR % nstate] = $1}
(NR % nstate == 0){
   mpd = pd / nstate;
   smpd = 0;
   for (i = 1; i <= nstate; i++) smpd += sd[i % nstate];
   rmpd = int((smpd + mpd) / 2 + 0.5);
   # Normal phones
   if (int(sd[0] + 0.5) == 0) {
      for (i = 1; i <= 3; i++) {
         sd[i % nstate] = int(sd[i % nstate] / smpd * rmpd + 0.5);
      }
      if (sd[3] <= 0) sd[3] = 1;
      for (i = 4; i <= nstate; i++) sd[i % nstate] = 0;
   }
   # Silence phone
   else {
      for (i = 1; i <= nstate; i++) {
          sd[i % nstate] = int(sd[i % nstate] / smpd * rmpd + 0.5);
      }
      if (sd[0] <= 0) sd[0] = 1;
   }
   if (sd[1] <= 0) sd[1] = 1;
   smpd = 0;
   for (i = 1; i <= nstate; i++) smpd += sd[i % nstate];
   for (i = 1; i <= nstate; i++) {
        if (sd[i % nstate] > 0) {
           tend = tstart + sd[i % nstate] * 50000;
           print tstart, tend, int(NR / nstate), i-1;
           tstart = tend;
        }
   }
   pd = 0;
}' 
done) > $datadir/synth_lab.mlf
# 3. Turn them into DNN input labels (i.e. one sample per frame)
python local/make_fullctx_mlf_dnn.py --extra-feats="$extra_feats" $datadir/synth_lab.mlf $datadir/cex.ark $datadir/feat.ark
copy-feats ark:$datadir/feat.ark ark,scp:$datadir/in_feats.ark,$datadir/in_feats.scp

lbldir=$datadir/lbl
mkdir -p $lbldir
cp $datadir/in_feats.scp $lbldir/feats.scp
cut -d ' ' -f 1 $lbldir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $lbldir/utt2spk
utils/utt2spk_to_spk2utt.pl $lbldir/utt2spk > $lbldir/spk2utt
#    steps/compute_cmvn_stats.sh $dir $dir $dir

# 4. Forward pass through pitch DNN
echo -e "\n** Pitch forward **\n"
pitchdir=$datadir/pitchout
local/make_forward_fmllr.sh --single $f0dnndir $lbldir $pitchdir ""

pitchlbldir=$datadir/pitchlbl
mkdir -p $pitchlbldir

# 5.a without mlpg
rm -rf $outdir/*
select-feats 0-1 ark:$pitchdir/feats.ark ark:- \
    | paste-feats ark:- scp:$lbldir/feats.scp ark,scp:$pitchlbldir/feats.ark,$pitchlbldir/feats.scp
cp $lbldir/{spk2utt,utt2spk} $pitchlbldir
echo -e "\n** Acoustic forward no mlpg **\n"
local/make_forward_fmllr.sh --single $dnndir $pitchlbldir $outdir ""
paste-feats ark:$pitchdir/feats.ark scp:$outdir/feats.scp ark,t:- \
    | awk -v dir=$outdir/cmp/ '($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'
mkdir -p $outdir/wav_nomlpg/; for cmp in $outdir/cmp/*.cmp; do
    local/mlsa_synthesis_pitch_mlpg.sh --synth $synth --voice_thresh $voice_thresh --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $mcep_order --delta_order $delta_order $cmp $outdir/wav_nomlpg/`basename $cmp .cmp`.wav
done

# 5b. mlpg on pitch values
# NB: rather evil bit of code that processes a kaldi feature
# file "in-line" with tools that only work on individual files
f0_win="-d win/logF0_d1.win -d win/logF0_d2.win"
varf0=`cat $var_cmp | awk '{printf "%f ", $2}' | cut -d " " -f 1-6`
#nfile=`cat $pitchdir/feats.scp | wc -l`
cat $pitchdir/feats.scp |\
while read line; do
    echo ${line%% *} "["
    copy-feats scp:"echo $line |" ark,t:- \
        | awk -v var="$varf0" '{ if ($NF == "]" || NR == 2) cvar = "0.0 0.0 0.0 0.0 0.0 0.0"; else cvar = var; if ($2 == "[") {} else {if ($NF == "]") {$0 = substr($0, 1, length($0) - 2);} print $0, cvar}}' \
        | x2x +a +f \
        | mlpg -i 0 -m 1 $f0_win \
        | x2x +f +a2
    echo "]"
done > $pitchdir/feats_mlpg.ark


# We have to recreate a kaldi pitch file
select-feats 0-1 ark:$pitchdir/feats_mlpg.ark ark:- \
    | paste-feats ark:- scp:$lbldir/feats.scp ark,scp:$pitchlbldir/feats.ark,$pitchlbldir/feats.scp

# 6. Forward pass through acoustic DNN

#pitchlbldir=$datadir/pitchlbl
#mkdir -p $pitchlbldir



local/make_forward_fmllr.sh --single $dnndir $pitchlbldir $outdir ""
paste-feats ark:$pitchdir/feats_mlpg.ark scp:$outdir/feats.scp ark,t:- \
    | awk -v dir=$outdir/cmp/ '($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'


# 7. Vocoding
mkdir -p $outdir/wav/; for cmp in $outdir/cmp/*.cmp; do
    local/mlsa_synthesis_pitch_mlpg.sh --mlpgf0done true --synth $synth --voice_thresh $voice_thresh --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $mcep_order --delta_order $delta_order $cmp $outdir/wav/`basename $cmp .cmp`.wav #$var_cmp
done

mkdir -p $outdir/wav_mlpg/; for cmp in $outdir/cmp/*.cmp; do
    local/mlsa_synthesis_pitch_mlpg.sh --mlpgf0done true --synth $synth --voice_thresh $voice_thresh --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $mcep_order --delta_order $delta_order $cmp $outdir/wav_mlpg/`basename $cmp .cmp`.wav $var_cmp
done

echo "Done. Samples are in $outdir/wav_mlpg/"

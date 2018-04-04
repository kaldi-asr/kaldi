durdnndir=exp_dnn/tts_dnn_dur_3_delta_quin5/
dnndir=exp_dnn/tts_dnn_train_3_deltasc2_quin5/
datadir=data/eval_test
spk=slt
srate=48000
delta_order=2
mcep_order=60
bndap_order=25
voice_thresh=0.5
alpha=0.55
fftlen=4096

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

rm -rf $datadir
mkdir -p $datadir
awk 'BEGIN{print "<parent>"}{print}END{print "</parent>"}' > $datadir/text.xml

# Generate CEX features for test set.
tpdb=$KALDI_ROOT/idlak-data/en/ga/
idlaktxp --pretty --tpdb=$tpdb $datadir/text.xml - \
    | idlakcex --pretty --cex-arch=default --tpdb=$tpdb - $datadir/text_full.xml
python local/idlak_make_lang.py --mode 2 -r "test" \
    $datadir/text_full.xml data/full/cex.ark.freq $datadir/cex.ark > $datadir/cex_output_dump
# Generate input feature for duration modelling
cat $datadir/cex.ark \
    | awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) for (state = 0; state < 5; state++) print a[i], state; print "]"}' \
    | copy-feats ark:- ark,scp:$datadir/in_durfeats.ark,$datadir/in_durfeats.scp

# Duration based test set
lbldurdir=lbldur$datadir
mkdir -p $lbldurdir
cp $datadir/in_durfeats.scp $lbldurdir/feats.scp
cut -d ' ' -f 1 $lbldurdir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $lbldurdir/utt2spk
utils/utt2spk_to_spk2utt.pl $lbldurdir/utt2spk > $lbldurdir/spk2utt
#steps/compute_cmvn_stats.sh $lbldurdir $lbldurdir $lbldurdir

# Generate label with DNN-generated duration

#  1. forward pass through duration DNN
duroutdir=$durdnndir/tst_forward_tmp/
rm -rf $duroutdir
utils/make_forward_fmllr.sh $durdnndir $lbldurdir $duroutdir ""
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
           print tstart, tend, int(NR / 5), i-1;
           tstart = tend;
        }
   }
   pd = 0;
}' 
done) > $datadir/synth_lab.mlf
# 3. Turn them into DNN input labels (i.e. one sample per frame)
python utils/make_fullctx_mlf_dnn.py $datadir/synth_lab.mlf $datadir/cex.ark $datadir/feat.ark
copy-feats ark:$datadir/feat.ark ark,scp:$datadir/in_feats.ark,$datadir/in_feats.scp

lbldir=lbl$datadir
mkdir -p $lbldir
cp $datadir/in_feats.scp $lbldir/feats.scp
cut -d ' ' -f 1 $lbldir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $lbldir/utt2spk
utils/utt2spk_to_spk2utt.pl $lbldir/utt2spk > $lbldir/spk2utt
#    steps/compute_cmvn_stats.sh $dir $dir $dir

# 4. Forward pass through big DNN
outdir=$dnndir/tst_forward_tmp/
rm -rf $outdir
utils/make_forward_fmllr.sh $dnndir $lbldir $outdir ""

# 5. Vocoding
# NB: these are the settings for 48k
mkdir -p $outdir/wav_mlpg/; for cmp in $outdir/cmp/*.cmp; do
    utils/mlsa_synthesis_63_mlpg.sh --voice_thresh $voice_thresh --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $mcep_order --delta_order $delta_order $cmp $outdir/wav_mlpg/`basename $cmp .cmp`.wav data/train/var_cmp.txt
done

echo "Done. Samples are in $outdir/wav_mlpg/"

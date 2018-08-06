cex_freq=data/full/cex.ark.freq
var_cmp=data/train/var_cmp.txt
durdnndir=exp_dnn/tts_dnn_dur_3_delta_quin5/
acsdnndir=exp_dnn/tts_dnn_train_3_deltasc2_quin5/
datadir=data/eval_test
spk=bdl
tpdbvar=en/ga/
srate=16000
delta_order=2
mcep_order=39
bndap_order=21
voice_thresh=0.5
alpha=0.42
fftlen=512

# NB: important input files are:
# data/full/cex.ark.freq (for mapping categorical features to binary)
# $durdnndir/{final.nnet,input_final.feature_transform,indelta_order,cmvn_glob.ark,final.feature_transform,norm_vars,cmvn_out_glob.ark}
# $dnndir/{final.nnet,input_final.feature_transform,indelta_order,cmvn_glob.ark,final.feature_transform,norm_vars,cmvn_out_glob.ark}
# tpdb dir
# data/train/var_cmp.txt

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

outputdir=${spk}_mdl
rm -rf $outputdir
mkdir -p $outputdir/{dur,acoustic,lang}
tpdbdir=$KALDI_ROOT/idlak-data/`dirname $tpdbvar`

for step in dur acoustic; do
    case $step in
        dur)
            dnndir=$durdnndir
            ;;
        acoustic)
            dnndir=$acsdnndir
            ;;
    esac
    # Make binary copies of nnet / transforms
    for i in $dnndir/{final.nnet,input_final.feature_transform}; do
        if [ -e "$i" ]; then
            nnet-copy $i $outputdir/$step/`basename $i`;
        fi
    done
    # Special case to reverse the output feature transform
    cat $dnndir/final.feature_transform \
        | utils/convert_transform.sh \
        | nnet-copy - $outputdir/$step/reverse_final.feature_transform
    # Turn features into binary as well
    for i in $dnndir/{incmvn_glob.ark,cmvn_out_glob.ark}; do
        if [ -e "$i" ]; then
            copy-feats $i $outputdir/$step/`basename $i`;
        fi
    done
    # Copy other files (they should really be merged into a configuration file)
    for i in $dnndir/{indelta_opts,delta_opts,cmvn_opts,incmvn_opts}; do
        if [ -e "$i" ]; then
            cp $i $outputdir/$step/`basename $i`;
        fi
    done
done

# Lang directory
cp -r $tpdbdir $outputdir/lang/
touch $outputdir/lang/idlak-data-trunk
cp $cex_freq $outputdir/lang/
cp $var_cmp $outputdir/lang/

# Config
for k in spk tpdbvar srate delta_order mcep_order bndap_order voice_thresh alpha fftlen; do
    echo "$k=${!k}"
done > $outputdir/voice.conf

echo 'All done! voice is in'  $outputdir

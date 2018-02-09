synth=cere # or excitation, cere

# Main settings
period=5
srate=16000
delta_order=0
fftlen=1024

# Acoustic features dimension
mcep_order=39  # 60
bndap_order=21 # 25
lsf_order=14   # 46
resid_order=0  # 30
f0_order=2     #  2

# Vocoder settings
voice_thresh=0.8
alpha=0.42

tmpdir=`mktemp -d`
win=win

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

cmp_file=$1
out_wav=$2
var_file=$3
base=`basename $cmp_file .cmp`

cmp=$tmpdir/$base.cmp
mcep=$tmpdir/$base.mcep
f0=$tmpdir/$base.f0
bap=$tmpdir/$base.bndap
lsf=$tmpdir/$base.blsf
resd=$tmpdir/$base.fpcagex

#pdf files
lpdf=$tmpdir/$base.blsf.pdf
rpdf=$tmpdir/$base.fpcagex.pdf
mpdf=$tmpdir/$base.mcep.pdf
fpdf=$tmpdir/$base.f0.pdf
bpdf=$tmpdir/$base.bndap.pdf
var=$tmpdir/$base.var

# dimension setup
numfeat=5
fshift=0.005
delta_mult=$(( $delta_order + 1 ))
f0_dim=$f0_order
mcep_dim=$(( $mcep_order + 1 ))
bndap_dim=$bndap_order
lsf_dim=$(( $lsf_order + 1 ))
resid_dim=$resid_order

# kaldi has added delta over the whole vector, so we need to put the deltas back to the right feature
# for mlpg
inblocks=$f0_dim:$mcep_dim:$lsf_dim:$resid_dim:$bndap_dim:$f0_dim:$mcep_dim:$lsf_dim:$resid_dim:$bndap_dim:$f0_dim:$mcep_dim:$lsf_dim:$resid_dim:$bndap_dim
#outblocks=0:5:10:1:6:11:2:7:12:3:8:13:4:9:14

# These should be something like: seq -s ":" $featid $numfeat $numfeat * $delta_mult
outf0=0:5:10
outmcep=1:6:11
outlsf=2:7:12
outresid=3:8:13
outbap=4:9:14
outblocks=$outf0:$outmcep:$outlsf:$outresid:$outbap
delta1="-0.2 -0.1 0 0.1 0.2"
delta2="0.285714 -0.142857 -0.285714 -0.142857 0.285714"
if [ $delta_order -eq 0 ]; then
    outf0=0
    outmcep=1
    outlsf=2
    outresid=3
    outbap=4
fi
cat data/train/var_cmp.txt | awk '{printf "%s ", $2}' > $var
utils/swap_file.py -i $inblocks -o $outblocks $var $var.2

utils/swap_file.py -i $inblocks -o $outblocks $cmp_file $cmp

utils/swap_file.py -i $inblocks -o $outf0    $cmp_file ${f0}c
utils/swap_file.py -i $inblocks -o $outmcep  $cmp_file $mcep
utils/swap_file.py -i $inblocks -o $outlsf   $cmp_file $lsf
utils/swap_file.py -i $inblocks -o $outresid $cmp_file $resd.d
utils/swap_file.py -i $inblocks -o $outbap   $cmp_file $bap.full

if [ $delta_order -gt 0 ]; then
    case $delta_order in
        1)
            #win="-d win/mcep_d1.win"
            win="-d $delta1"
            ;;
        2)
            #win="-d win/mcep_d1.win -d win/mcep_d2.win"
            win="-d $delta1 -d $delta2"
            ;;
    esac
    # Split files accordingly
    #delta_mult=$(( $delta_order + 1 ))
    f0_len=$(( $f0_dim * $delta_mult ))
    lsf_len=$(( $lsf_dim * $delta_mult ))
    mcep_len=$(( $mcep_dim * $delta_mult ))
    resd_len=$(( $resid_dim * $delta_mult ))
    bndap_len=$(( $bndap_dim * $delta_mult ))
    # Make pdf with unit variance
    cat ${f0}c | awk -v dim=$f0_dim -v len=$f0_len '{for (i=1; i<=len;i++) print $i; for (i=1; i<=len;i++) {v=0.003; if (i < 2 * dim) v=0.03; if (i < dim) v=1;  print v}}' | x2x +af > $fpdf
    cat $lsf | awk -v dim=$lsf_dim -v len=$lsf_len '{for (i=1; i<=len;i++) print $i; for (i=1; i<=len;i++) {v=0.003; if (i < 2 * dim) v=0.03; if (i < dim) v=1;  print v}}' | x2x +af > $lpdf
    cat $resd.d | awk -v dim=$resid_dim -v len=$resd_len '{for (i=1; i<=len;i++) print $i; for (i=1; i<=len;i++) {v=0.003; if (i < 2 * dim) v=0.03; if (i < dim) v=1;  print v}}' | x2x +af > $rpdf
    cat $mcep | awk -v dim=$mcep_dim -v len=$mcep_len '{for (i=1; i<=len;i++) print $i; for (i=1; i<=len;i++) {v=0.003; if (i < 2 * dim) v=0.03; if (i < dim) v=1;  print v}}' | x2x +af > $mpdf
    cat $bap.full | awk -v dim=$bndap_dim -v len=$bndap_len '{for (i=1; i<=len;i++) print $i; for (i=1; i<=len;i++) {v=0.003; if (i < 2 * dim) v=0.03; if (i < dim) v=1;  print v}}' | x2x +af > $bpdf
    # Run mlpg
    mlpg -i 0 -m $(( $f0_order - 1 )) $win $fpdf | x2x +f +a$f0_dim > ${f0}c
    mlpg -i 0 -m $mcep_order $win $mpdf | x2x +f +a$mcep_dim > $mcep
    mlpg -i 0 -m $lsf_order $win $lpdf | x2x +f +a$lsf_dim > $lsf
    mlpg -i 0 -m $(( $resid_order -1 )) $win $rpdf | x2x +f +a$resid_dim > $resd.d
    mlpg -i 0 -m $(( $bndap_order - 1 )) $win $bpdf | x2x +f +a$bndap_dim > $bap.full
fi
    

# Post processing
cat ${f0}c | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
cat $bap.full | awk '{for (i = 1; i <= NF; i++) if ($i >= 0.0) printf "0.0 " ; else printf "%f ", 20 * ($i) / log(10); printf "\n"}' > $bap
head -n 4 data/dev_slt_lsftmp/arctic_a0002.fpcagex > $resd
cat $resd.d | awk -v frate=$fshift '{print frate * NR, frate * (NR + 1), $0}' >>  $resd
cp data/dev_slt_lsftmp/arctic_a0002.pcagex.gres $tmpdir/$base.pcagex.gres

case $synth in
    cerepca)
        dspdir=$HOME/cereproc/trunk/apps/dsplab
        pcadir=$dspdir/pca_tools
        pca_opts="--bndap_order 21 --gres_threshold 0 --pm_ext=.pmrp --fft_order=1024 --residual_ext=.gex --pca_type 1 -s $srate"
        model=data/slt_pcamodel
        python $pcadir/fix_pca_frames.py -d -p $tmpdir -o $tmpdir -f $tmpdir -F $fshift $base
        #touch $tmpdir/$base.pcagex.gres
        python $pcadir/pca_vocoder.py $pca_opts -s $srate -n 1 -r "$base.*" --decode -i $tmpdir -o $tmpdir $model
        python $dspdir/vocodelsp.py --BL -s $srate -f $fftlen -F $fshift -m $lsf_order -E $tmpdir/$base.gex -M $lsf -o $tmpdir
        cp $tmpdir/$base.wav $out_wav
        ;;
    cerelsf)
        python $HOME/cereproc/trunk/apps/dsplab/vocodelsp.py -e -R -P -s $srate -f $fftlen -b $bndap_order -m $lsf_order --BL -K -i $tmpdir -o $tmpdir $base
        cp $tmpdir/$base.wav $out_wav
        ;;
    cere)
        if [ `awk -v vt=$voice_thresh 'BEGIN{if (vt > 0) print 1; else print 0}'` -eq 1 ]; then
            mlopts="-n"
        else
            mlopts="-p"
        fi
        cmd="python $HOME/cereproc/trunk/apps/dsplab/mlsa.py $mlopts -c -K -C -a $alpha -m $mcep_order -s $srate -f $fftlen -b $bndap_order -i $tmpdir -o $tmpdir -e $base"
        echo $cmd
        $cmd
        cp $tmpdir/$base.wav $out_wav
        ;;
esac


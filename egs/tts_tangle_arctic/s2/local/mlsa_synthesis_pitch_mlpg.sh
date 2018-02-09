#!/bin/bash
#
# Copyright 2016 by CereProc LTD
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Blaise Potard, 2016
#

# Allow setshell
#software=/idiap/resource/software
#source $software/initfiles/shrc $software
#win=/remote/lustre/1/temp/bpotard/src/kaldi-trunk/egs/TTS/win

# Perform raw resynthesis of cmp file
#SETSHELL straight
#SETSHELL sptk

#TMP=/scratch/$USER/tmp
#mkdir -p $TMP
synth=excitation # or excitation, cere
period=5
srate=16000
delta_order=0
mcep_order=39
bndap_order=21
f0_order=2
# Continuous f0, no mixing
mlpgf0done=false
voice_thresh=0.8
alpha=0.55
fftlen=1024
tmpdir=`mktemp -d`
win=win

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;


cmp_file=$1
out_wav=$2
var_file=$3
#var_file_pitch=$4
base=`basename $cmp_file .cmp`
var=$tmpdir/var
cmp=$tmpdir/$base.cmp
uv=$tmpdir/$base.uv
#variance file
vmcep=$tmpdir/$base.mcep.var
vf0=$tmpdir/$base.f0.var
vbap=$tmpdir/$base.bndap.var
#pdf files
mpdf=$tmpdir/$base.mcep.pdf
fpdf=$tmpdir/$base.f0.pdf
bpdf=$tmpdir/$base.bndap.pdf
resid=$tmpdir/$base.resid
#feature files
mcep=$tmpdir/$base.mcep
f0=$tmpdir/$base.f0
bap=$tmpdir/$base.bndap

apf=$tmpdir/$base.apf
sspec=$tmpdir/$base.spec

rm -f $var $cmp $uv $vmcep $vf0 $vbap $mpdf $fpdf $bpdf $mcep $f0 $bap $apf $sspec


#mcep_order=39
# Assuming: F0 - MCEP - BNDAP
delta_mult=$(( $delta_order + 1 ))
f0_offset=1
f0_vlen=$(( $f0_order * $delta_mult ))
if [ "$mlpgf0done" == "true" ]; then
    f0_len=$f0_order
else
    f0_len=$(( $f0_order * $delta_mult ))
fi

mcep_offset=$(( $f0_offset + $f0_len ))
mcep_voffset=$(( $f0_offset + $f0_vlen ))
mcep_len=$(( ($mcep_order + 1) * $delta_mult ))
order=$mcep_order

bndap_offset=$(( $mcep_offset + $mcep_len ))
bndap_voffset=$(( $mcep_voffset + $mcep_len ))
bndap_len=$(( $bndap_order * $delta_mult ))

echo $f0_offset $f0_len $mcep_offset $mcep_len $bndap_offset $bndap_len

f0_dim=$f0_order
mcep_dim=$(( $mcep_order + 1 ))
bndap_dim=$bndap_order

if [ "$mlpgf0done" == "true" ]; then
    inblocks=$f0_dim:0:0:$mcep_dim:$bndap_dim:$mcep_dim:$bndap_dim:$mcep_dim:$bndap_dim
else
    inblocks=$f0_dim:$f0_dim:$f0_dim:$mcep_dim:$bndap_dim:$mcep_dim:$bndap_dim:$mcep_dim:$bndap_dim
fi
outblocks=0:1:2:3:5:7:4:6:8

local/swap_file.py -i $inblocks -o $outblocks $cmp_file $cmp

if [ "$var_file" != "" ]; then
    if [ $delta_mult -eq 2 ]; then
	mcep_win="-d $win/mcep_d1.win"
	f0_win="-d $win/logF0_d1.win"
	bndap_win="-d $win/bndap_d1.win"
    elif [ $delta_mult -eq 3 ]; then
	mcep_win="-d $win/mcep_d1.win -d $win/mcep_d2.win"
	f0_win="-d $win/logF0_d1.win -d $win/logF0_d2.win"
	bndap_win="-d $win/bndap_d1.win -d $win/bndap_d2.win"
    else
	mcep_win=""
	f0_win=""
	bndap_win=""
    fi
    echo "Extracting variances..."
    cat $var_file | awk '{printf "%f ", $2}' \
        | local/swap_file.py \
        -i $f0_dim:$f0_dim:$f0_dim:$mcep_dim:$bndap_dim:$mcep_dim:$bndap_dim:$mcep_dim:$bndap_dim \
        -o 0:1:2:3:5:7:4:6:8 \
        -  $var

    cat $var | cut -d " " -f $mcep_voffset-$(( $mcep_voffset + $mcep_len - 1 )) > $vmcep
    cat $var | cut -d " " -f $f0_offset-$(( $f0_offset + $f0_vlen - 1 )) > $vf0
    cat $var | cut -d " " -f $bndap_voffset-$(( $bndap_voffset + $bndap_len - 1 )) > $vbap

    echo "Creating pdfs..."
    cat $cmp | cut -d " " -f $mcep_offset-$(( $mcep_offset + $mcep_len - 1 )) | awk -v var="`cat $vmcep`" '{print $0, var}' | x2x +a +f > $mpdf
    cat $cmp | cut -d " " -f $f0_offset-$(( $f0_offset + $f0_len - 1 )) | awk -v var="`cat $vf0`" '{print $0, var}' | x2x +a +f > $fpdf
    cat $cmp | cut -d " " -f $bndap_offset-$(( $bndap_offset + $bndap_len - 1 )) | awk -v var="`cat $vbap`" '{print $0, var}' | x2x +a +f > $bpdf
    
    echo "Running mlpg..."
    echo "mcep smoothing"
    mlpg -i 0 -m $mcep_order $mcep_win $mpdf | x2x +f +a$(( $mcep_order + 1 )) > $mcep
    if [ "$mlpgf0done" != true ]; then
        echo "f0 smoothing"
        mlpg -i 0 -m $(( $f0_order - 1 )) $f0_win $fpdf > ${f0}_raw
    else
        cat $cmp | cut -d " " -f $f0_offset-$(( $f0_offset + $f0_len - 1 )) | x2x +a +f > ${f0}_raw
    fi
    echo "bndap smoothing"
    mlpg -i 0 -m $(( $bndap_order - 1 )) $bndap_win $bpdf | x2x +f +a$bndap_order | awk '{for (i = 1; i <= NF; i++) if ($i >= -0.5) printf "0.0 " ; else printf "%f ", 20 * ($i + 0.5) / log(10); printf "\n"}' > $bap

    #cat $cmp | cut -d " " -f $(($f0_offset + 1))-$(($f0_offset + 1)) > $uv
    cat ${f0}_raw | x2x +f +a$f0_order | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
    paste $f0 $bap | awk -v FS='\t' -v n=$bndap_order 'BEGIN{zero="0"; for (i = 1; i < n; i++) zero = zero " 0"}{if ($1 > 0.0) print $2; else print zero}' > $bap.2
    mv $bap.2 $bap
    
    # Do not do mlpg on MCEP
    #cat $cmp | cut -d " " -f $mcep_offset-$(($mcep_offset + $mcep_order)) > $mcep
else
    cat $cmp | cut -d " " -f $mcep_offset-$(($mcep_offset + $mcep_order)) > $mcep
    cat $cmp | cut -d " " -f $f0_offset-$(($f0_offset + $f0_order - 1)) | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
    cat $cmp | cut -d " " -f $bndap_offset-$(($bndap_offset + $bndap_order - 1)) | awk '{for (i = 1; i <= NF; i++) if ($i >= -0.5) printf "0.0 " ; else printf "%f ", 20 * ($i+0.5) / log(10); printf "\n"}' > $bap
    paste $f0 $bap | awk -v FS='\t' -v n=$bndap_order 'BEGIN{zero="0"; for (i = 1; i < n; i++) zero = zero " 0"}{if ($1 > 0.0) print $2; else print zero}' > $bap.2
    mv $bap.2 $bap
fi

# if we want to keep the original f0/apf:
#cat $cmp_file | sed -e 's/^ *//g' -e 's/ *$//g' | cut -d " " -f $f0_offset-$(($f0_offset + 1)) | awk -v thresh=$voice_thresh '{if ($2 > thresh) print exp($1); else print 0.0}' > $f0
#cat $cmp_file | sed -e 's/^ *//g' -e 's/ *$//g' | cut -d " " -f $bndap_offset-$(($bndap_offset + $bndap_order - 1)) | x2x +a +f > $bap

#echo "synthesis_fft -mel -bap -order $mcep_order -apfile $bap $f0 $mcep $2"
#/idiap/user/pehonnet/HTS-ENGINE-for-HTS-2.1/
#synthesis_fft -float -f $samp_freq -sigp 1.2 -cornf 1000 -bw 70.0 -delfrac 0.2 -sd 0.5 -mel -bap -order $mcep_order -apfile $bap -alpha $alpha $f0 $mcep $2

if [ "$synth" = "cere" ]; then
    if [ `awk -v vt=$voice_thresh 'BEGIN{if (vt > 0) print 1; else print 0}'` -eq 1 ]; then
        mlopts="-n -c"
    else
        mlopts="-p"
    fi
    if [ "$srate" = "48000" ]; then
        mlopts="$mlopts -B 8,15,22,30,38,47,58,69,81,95,110,127,147,169,194,224,259,301,353,416,498,606,757,980,1344"
    fi
    cmd="python $HOME/cereproc/trunk/apps/dsplab/mlsa.py $mlopts -C -a $alpha -m $order -s $srate -f $fftlen -b $bndap_order -i $tmpdir -o $tmpdir $base"
    echo $cmd
    $cmd
    cp $tmpdir/$base.wav $out_wav
elif [ "$synth" = "excitation" ]; then
    x2x +af $mcep > $mcep.float
    psize=`echo "$period * $srate / 1000" | bc`
    # We have to drop the first few F0 frames to match SPTK behaviour
    #cat $f0 | awk -v srate=$srate '(NR > 2){if ($1 > 0) print srate / $1; else print 0.0}' | x2x +af \
    #    | excite -p $psize \
    python local/excitation.py -s $srate -f $fftlen -b $bndap_order $f0 $bap > $tmpdir/resid.float
    cat $tmpdir/resid.float | mlsadf -P 5 -m $order -a $alpha -p $psize $mcep.float | x2x -o +fs > $tmpdir/data.mcep.syn
    sox --norm -t raw -c 1 -r $srate -s -b 16 $tmpdir/data.mcep.syn $out_wav
elif [ "$synth" = "convolve" ]; then
    x2x +af $mcep > $mcep.float
    psize=`echo "$period * $srate / 1000" | bc`
    isize=`echo "$period * $srate / 5000" | bc`
    echo "Excitation"
    python local/excitation.py -s $srate -f $fftlen -b $bndap_order $f0 $bap $resid.float #$bap
    # Generate mcep spectogram; note that phase has been lost so output waveform will look very different
    echo "Spectrum $fftlen"
    # Rather slow process: we have to generate amplitude and phase separately
    mgc2sp -l $fftlen -m $order -a $alpha -o 2 $mcep.float > $mcep.sp.norm.float
    mgc2sp -l $fftlen -m $order -a $alpha -p -o 1 $mcep.float > $mcep.sp.arg.float
    # Combine norm and angle
    merge -s 1 -l 1 -L 1 +f $mcep.sp.arg.float < $mcep.sp.norm.float > $mcep.sp
    echo "Convolution $mcep.sp"
    python local/convolve.py -m 4.0 -s -l $fftlen -p $psize -i $isize $resid.float $mcep.sp $out_wav
elif [ "$synth" = "WORLD" ]; then
    x2x +ad $bap > $bap.double
    x2x +af $mcep > $mcep.float
    x2x +ad $f0 > $f0.double
    world=/home/pilar/Documents/idlak-merlin/dnn_tts/tools/WORLD/build
    mgc2sp -l $fftlen -g 0 -m $order -a $alpha -o 2 $mcep.float | sopr -d 55000.0 -P | x2x +fd > $mcep.sp.double
    echo $world/synth $fftlen $srate $f0.double $mcep.sp.double $bap.double $out_wav
    $world/synth $fftlen $srate $f0.double $mcep.sp.double $bap.double $out_wav
else
    x2x +af $mcep > $mcep.float
    psize=`echo "$period * $srate / 1000" | bc`
    # We have to drop the first few F0 frames to match SPTK behaviour
    cat $f0 | awk -v srate=$srate '(NR > 2){if ($1 > 0) print srate / $1; else print 0.0}' | x2x +af \
        | excite -p $psize \
        | mlsadf -P 5 -m $order -a $alpha -p $psize $mcep.float | x2x -o +fs > $tmpdir/data.mcep.syn
    sox --norm -t raw -c 1 -r $srate -s -b 16 $tmpdir/data.mcep.syn $out_wav
fi

#-sigp 1.2
#          -sd 0.5
#          -cornf 100
#          -bw 70.0
#          -delfrac 0.2
# rm -f $mcep $f0 $bap $apf $sspec
#synthesis_fft -f $samp_freq -mel -order $mcep_order $f0 $mcep $2
#synthesis_fft -mel -bap -order 40 -float $f0 $mcep /tmp/out.wav -apfile $apf

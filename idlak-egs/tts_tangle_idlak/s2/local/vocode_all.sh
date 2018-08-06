#!/bin/bash
#
# Copyright 2013 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Blaise Potard, 2013-2017
#

help_message="
Usage: vocode_all.sh [options] <wav_in> <wav_out>
Main options:
  --synth <type>     # The type of vocoding to use. Available choices:
                     # idlak, sptk, cere, straight, world, idlak+straight, idlak+world
  --srate <srate>    # Sampling rate of audio file
  --mcep-order   <n> # Order of mcep extraction
  --bndap-order  <n> # Order of bndap extraction
  --voice-thresh <f> # F0 thresholding for continuous pitch extraction, by default -1
  --alpha        <f> # Warping coefficient for mcep, default 0.55
  --fftlen       <n> # FFT length, default 1024"

synth=idlak # choice of: idlak, sptk, cere, straight, world
period=5 # in ms
srate=48000 # in Hz
mcep_order=59
bndap_order=5
#f0_order=2
voice_thresh=0.5 # Low voicing is turned into no voicing
alpha=0.55
fftlen=2048
tmpdir=/tmp

# Tools directory:
idlak=/home/potard/aria/idlak-git/src/featbin
kaldi=/home/potard/aria/kaldi-orig/src/featbin
world=/home/pilar/Documents/idlak-merlin/dnn_tts/tools/WORLD/build
straight=/home/potard/cereproc/trunk/tools/linux_x86_64/hts_tools/0912/bin
sptk=/home/potard/aria/idlak-git/tools/SPTK/bin
cere=/home/potard/cereproc/trunk/apps/dsplab

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

wav=$1
out_wav=$2
base=`basename $wav .wav`
mcep=$tmpdir/$base.mcep
raw=$tmpdir/$base.raw
f0=$tmpdir/$base.f0
sp=$tmpdir/$base.sp
bap=$tmpdir/$base.bndap
resid=$tmpdir/$base.resid
mcep_len=$(( $mcep_order + 1 ))

case $synth in
    cere)
        # Analyse with straight, synthesize with out own tools
        # NB: this does not sound nearly as good as straight with the parameters,
        # so we are obviously doing something wrong!
        $straight/tempo -maxf0 300 -shift 1 -f0shift $period $wav $f0
        $straight/straight_mcep -pow -float -f0shift $period -shift $period -fftl $fftlen -f0file $f0 -order $mcep_order -alpha $alpha -mcep $wav $mcep.float
        $straight/straight_bndap -float -f0shift $period -shift $period -fftl $fftlen -f0file $f0 -bndap $wav $bap.float
        # Synthesis
        $sptk/x2x +f +a$mcep_len $mcep.float > $mcep
        $sptk/x2x +f +a5 $bap.float > $bap
        $cere/mlsa.py -c -C -a $alpha -m $mcep_order -s $srate -f $fftlen -b 5 -i $tmpdir -o $tmpdir -e $base
        cp $tmpdir/$base.wav $out_wav
        ;;
    straight)
        # Analyse and synthese with straight
        # Should we try a lower period?
        $straight/tempo -maxf0 300 -shift 1 -f0shift 1 $wav $f0
        $straight/straight_mcep -pow -float -f0shift 1 -shift 1 -fftl $fftlen -f0file $f0 -order $mcep_order -alpha $alpha -mcep $wav $mcep.float
        $straight/straight_bndap -float -f0shift 1 -shift 1 -fftl $fftlen -f0file $f0 -bndap $wav $bap.float
        # Decimate mcep / bndap / f0
        $sptk/decimate -p $period -l $(( $mcep_order + 1 )) $mcep.float > $mcep.float.dc
        $sptk/decimate -p $period -l 5 $bap.float > $bap.float.dc
        $sptk/x2x +af $f0 | $sptk/decimate -p $period -l 1 | $sptk/x2x +fa1 >  $f0.dc
        # Synthesis
        $straight/synthesis_fft -shift $period -float -alpha $alpha -mel -order $mcep_order -f $srate -fftl $fftlen -apfile $bap.float.dc $f0.dc $mcep.float.dc $out_wav
        ;;
    sptk)
        # Analyse and synthesis with sptk
        sk=`echo "$srate / 1000" | bc`
        psize=`echo "$period * $srate / 1000" | bc`
        sox $wav -t raw - | $sptk/x2x +sf > $raw.float # or sox -t raw | x2x +sf ?
        # F0 generation
        $sptk/pitch -a 1 -s $sk -p $psize -L 60 -H 300 -o 1 $raw.float > $f0.float
        # TODO: calculate frame_length from min_f0
        minf0=`sopr -magic 0 $f0.float | minmax -o 1 | x2x +fa`
        # winlen in samples
        winlen=`sopr -magic 0 $f0.float | minmax -o 1 | sopr -INV -m $srate -m 2.3 -FIX | x2x +fa`
        #winlen=$(( $srate * $frame_length / 1000 ))
        # Mcep generation
        $sptk/frame -l $winlen -p $psize < $raw.float | $sptk/window -l $winlen -L $fftlen \
            | $sptk/fftr -l $fftlen -A -H | $sptk/mcep -e 1.0e-8 -f 0.0 -j 100 -l $fftlen -m $mcep_order -a $alpha -q 3 > $mcep.float
        # NO bndap :-)
        # Synthesis
        $sptk/sopr -magic 0 -INV -m $srate -MAGIC 0 $f0.float | $sptk/excite -p $psize \
            | $sptk/mlsadf -P 5 -m $mcep_order -a $alpha -p $psize $mcep.float | $sptk/x2x -o +fs > $out_wav.syn 2> /dev/null
        sox --norm -t raw -c 1 -r $srate -e signed-integer -b 16 $out_wav.syn $out_wav
        ;;
    world)
        # WORLD analysis
        fftlen=2048
        $world/analysis $wav $f0.double $sp.double $bap.double
        # Convert world spectrum to mcep
        $sptk/x2x +df $sp.double | $sptk/sopr -R -m 32768.0 | $sptk/mcep -a $alpha -m $mcep_order -l $fftlen -e 1.0E-8 -j 0 -f 0.0 -q 3 > $mcep.float
        # Convert back to spectrum
        $sptk/mgc2sp -a $alpha -g 0 -m $mcep_order -l $fftlen -o 2 $mcep.float | $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > $sp.resyn.double
        # WORLD synthesis
        $world/synth $fftlen $srate $f0.double $sp.resyn.double $bap.double $out_wav
        ;;
    idlak)
        input="scp:echo a $wav |"
        inputf0="scp:echo a $f0 |"
        output="ark,t:| tail -n +2 | awk '{if (\$NF == \"]\") \$NF=\"\"; print}'"
        common_opts="--snip-edges=false --frame-shift=$period --sample-frequency=$srate"
        f0fl=100
        psize=`echo "$period * $srate / 1000" | bc`
        nd=$(( $f0fl / $period ))
        $kaldi/compute-kaldi-pitch-feats $common_opts --frame-length=$f0fl "$input" ark:$f0.ark #"$output" > $f0
        $idlak/copy-feats ark:$f0.ark "$output" | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
        $idlak/copy-feats ark:$f0.ark "$output" | awk -v thresh=$voice_thresh '{print NR * 0.005, $1}' > $f0.prob
        $idlak/compute-aperiodic-feats $common_opts --max-iters=1 --use-hts-bands=true --num-mel-bins=$bndap_order "$input" ark:$f0.ark "$output" > $bap 2> /dev/null
        sox $wav -t raw - | $sptk/x2x +sf > $raw.float # or sox -t raw | x2x +sf ?
        winlen=`x2x +af $f0 | sopr -magic 0 | minmax -o 1 | sopr -INV -m $srate -m 2.3 -FIX | x2x +fa`
        $sptk/frame -l $winlen -p $psize < $raw.float | $sptk/window -l $winlen -L $fftlen \
            | $sptk/fftr -l $fftlen -A -H | $sptk/mcep -e 1.0e-8 -f 0.0 -j 0 -l $fftlen -m $mcep_order -a $alpha -q 3 > $mcep.float
        cat $f0 | awk -v nd=$nd '(NR > nd){print}END{for (i = 0; i <= nd; i++) print 0.0}' > $f0.txt
        cat $bap | awk '{for (i = 1; i <= NF; i++) if ($i >= 0.0) printf "0.0 " ; else printf "%f ", 20 * ($i) / log(10); printf "\n"}' > $bap.2
        #paste $f0.txt - | awk -v FS='\t' -v n=$bndap_order 'BEGIN{zero="0"; for (i = 1; i < n; i++) zero = zero " 0"}{if ($1 > 0.0) print $2; else print zero}' > $bap.2
        mv $bap.2 $bap
        python utils/excitation.py -s $srate -f $fftlen -b $bndap_order $f0.txt $bap $resid.float
        # Generate mcep spectogram; note that phase has been lost so output waveform will look very different
        echo "Spectrum $fftlen"
        # Rather slow process: we have to generate amplitude and phase separately
        mgc2sp -l $fftlen -m $mcep_order -a $alpha -o 2 $mcep.float > $mcep.sp.norm.float
        mgc2sp -l $fftlen -m $mcep_order -a $alpha -p -o 1 $mcep.float > $mcep.sp.arg.float
        # Combine norm and angle
        merge -s 1 -l 1 -L 1 +f $mcep.sp.arg.float < $mcep.sp.norm.float > $mcep.sp
        echo "Convolution $mcep.sp"
        isize=`echo "$period * $srate / 5000" | bc`
        python $HOME/cereproc/trunk/apps/dsplab/convolve.py -r $srate -s -l $fftlen -p $psize -i $isize $resid.float $mcep.sp $out_wav
        ;;
    idlak+straight)
        input="scp:echo a $wav |"
        inputf0="ark:cat $f0.dc | awk 'BEGIN{print \"a [\"}{if (\$1 == 0.0) print \"0.0 0.0\"; else print \"1.0\", \$1}END{print \"]\"}' |" 
        output="ark,t:| tail -n +2 | awk '{if (\$NF == \"]\") \$NF=\"\"; print}'"
        common_opts="--snip-edges=false --frame-shift=$period --sample-frequency=$srate"
        f0fl=60
        nd=$(( $f0fl / $period ))
        #$kaldi/compute-kaldi-pitch-feats $common_opts --frame-length=$f0fl "$input" ark:$f0.ark #"$output" > $f0
        #$idlak/copy-feats ark:$f0.ark "$output" | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
        #$idlak/copy-feats ark:$f0.ark "$output" | awk -v thresh=$voice_thresh '{print NR * 0.005, $1}' > $f0.prob
        # TODO: get bands information and use it in synthesis
        
        $straight/tempo -maxf0 300 -shift 1 -f0shift 1 $wav $f0.s
        $sptk/x2x +af $f0.s | $sptk/decimate -p $period -l 1 | $sptk/x2x +fa1 >  $f0.dc
        $straight/straight_mcep -pow -float -f0shift 1 -shift 1 -fftl $fftlen -f0file $f0.s -order $mcep_order -alpha $alpha -mcep $wav $mcep.float
        $sptk/decimate -p $period -l $(( $mcep_order + 1 )) $mcep.float > $mcep.float.dc
        #echo $idlak/compute-aperiodic-feats $common_opts --max-iters=1 --use-hts-bands=true --num-mel-bins=$bndap_order "$input" "$inputf0" "$output"
        $idlak/compute-aperiodic-feats $common_opts --max-iters=1 --use-hts-bands=true --num-mel-bins=$bndap_order "$input" "$inputf0" "$output" > $bap 2> /dev/null
        # Synthesis
        #cat $f0 | awk -v nd=$nd '(NR > nd){print}END{for (i = 0; i <= nd; i++) print 0.0}' > $f0.txt
        cat $bap | awk '{for (i = 1; i <= NF; i++) if ($i >= 0.0) printf "0.0 " ; else printf "%f ", 20 * ($i) / log(10); printf "\n"}' > $bap.2
        #paste $f0.txt - | awk -v FS='\t' -v n=$bndap_order 'BEGIN{zero="0"; for (i = 1; i < n; i++) zero = zero " 0"}{if ($1 > 0.0) print $2; else print zero}' > $bap.2
        mv $bap.2 $bap
        python utils/excitation.py -s $srate -f $fftlen -b $bndap_order $f0.dc $bap $resid.float
        # Generate mcep spectogram; note that phase has been lost so output waveform will look very different
        echo "Spectrum $fftlen"
        # Rather slow process: we have to generate amplitude and phase separately
        mgc2sp -l $fftlen -m $mcep_order -a $alpha -o 2 $mcep.float.dc > $mcep.sp.norm.float
        mgc2sp -l $fftlen -m $mcep_order -a $alpha -p -o 1 $mcep.float.dc > $mcep.sp.arg.float
        # Combine norm and angle
        merge -s 1 -l 1 -L 1 +f $mcep.sp.arg.float < $mcep.sp.norm.float > $mcep.sp
        echo "Convolution $mcep.sp"
        psize=`echo "$period * $srate / 1000" | bc`
        isize=`echo "$period * $srate / 5000" | bc`
        python $HOME/cereproc/trunk/apps/dsplab/convolve.py -r $srate -m 4.0 -s -l $fftlen -p $psize -i $isize $resid.float $mcep.sp $out_wav

        ;;
    idlak+world)
        fftlen=2048
        input="scp:echo a $wav |"
        inputf0="scp:echo a $f0 |"
        output="ark,t:| tail -n +2 | awk '{if (\$NF == \"]\") \$NF=\"\"; print}'"
        common_opts="--frame-shift=$period --snip-edges=false --sample-frequency=$srate"
        f0fl=60
        nd=$(( $f0fl / $period ))
        $kaldi/compute-kaldi-pitch-feats $common_opts --frame-length=$f0fl "$input" ark:$f0.ark #"$output" > $f0
        $idlak/copy-feats ark:$f0.ark "$output" | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
        $idlak/copy-feats ark:$f0.ark "$output" | awk -v thresh=$voice_thresh '{print NR * 0.005, $1}' > $f0.prob
        # TODO: get bands information and use it in synthesis
        $idlak/compute-aperiodic-feats $common_opts --max-iters=1 --use-hts-bands=true --num-mel-bins=$bndap_order "$input" ark:$f0.ark "$output" > $bap 2> /dev/null
        $world/analysis $wav $f0.double $sp.double $bap.double
        # Convert world spectrum to mcep
        $sptk/x2x +df $sp.double | $sptk/sopr -R -m 32768.0 | $sptk/mcep -a $alpha -m $mcep_order -l $fftlen -e 1.0E-8 -j 0 -f 0.0 -q 3 > $mcep.float
        
        # Synthesis
        #rm -f $bap.double
        # For some weird reasons we do not use the same convention for bndap
        cat $f0 | awk -v nd=$nd '(NR > nd){print}END{for (i = 0; i <= nd; i++) print 0.0}' > $f0.txt
        cat $bap | awk '{for (i = 1; i <= NF; i++) if ($i >= 0.0) printf "0.0 " ; else printf "%f ", 20 * ($i) / log(10); printf "\n"}' > $bap.2
        #paste $f0.txt - | awk -v FS='\t' -v n=$bndap_order 'BEGIN{zero="0"; for (i = 1; i < n; i++) zero = zero " 0"}{if ($1 > 0.0) print $2; else print zero}' > $bap.2
        mv $bap.2 $bap
        python utils/excitation.py -s $srate -f $fftlen -b $bndap_order $f0.txt $bap $resid.float
        # Generate mcep spectogram; note that phase has been lost so output waveform will look very different
        echo "Spectrum $fftlen"
        # Rather slow process: we have to generate amplitude and phase separately
        mgc2sp -l $fftlen -m $mcep_order -a $alpha -o 2 $mcep.float > $mcep.sp.norm.float
        mgc2sp -l $fftlen -m $mcep_order -a $alpha -p -o 1 $mcep.float > $mcep.sp.arg.float
        # Combine norm and angle
        merge -s 1 -l 1 -L 1 +f $mcep.sp.arg.float < $mcep.sp.norm.float > $mcep.sp
        echo "Convolution $mcep.sp"
        psize=`echo "$period * $srate / 1000" | bc`
        isize=`echo "$period * $srate / 5000" | bc`
        python $HOME/cereproc/trunk/apps/dsplab/convolve.py -r $srate -s -l $fftlen -p $psize -i $isize $resid.float $mcep.sp $out_wav
        ;;
esac
# Silent assassin
#rm -f $tmpdir/${base}.*
    
    

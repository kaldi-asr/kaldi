#!/usr/bin/env bash

# testing the graph creation (verifies we get the same answer as with HTK)
# needs the model-list.
# Call with model dir: convert_basic or convert_factored

dir=convert_basic
mlist=/mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/xwrd.clustered.mlist
mmf=/mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm164/MMF


rootdir=../..
D=`dirname "$rootdir"`
B=`basename "$rootdir"`
rootdir="`cd \"$D\" 2>/dev/null && pwd || echo \"$D\"`/$B"
export PATH=$PATH:$rootdir/tools/openfst/bin/:$rootdir/src/fstbin/:$rootdir/src/bin/


cd $dir

mkdir -p test
fstrandgen --select=log_prob CLG.fst   > test/fst.test
cat test/fst.test | fstprint --isymbols=context_syms.txt --osymbols=words.txt > test/fst.txt
cat test/fst.txt | awk ' { print $3; } ' | sed 's:<eps>:sil:g' | perl -ane 'if(m:(.+)/(.+)/(.+):) {  if($2 eq "sil") { print "$2\n"; } else { print "$1-$2+$3\n"; } } ' > test/xword.seq


cat test/xword.seq | perl -e 'open(F, "<$ARGV[0]")||die "Died opening model list."; 
  while(<F>){ @A=split(" ",$_); if(@A==1){ $phys{$A[0]}=$A[0];} else { $phys{$A[0]}=$A[1];} }
  while(<STDIN>) { @A=split(" ", $_); if(@A>0) { @A==1||die "bad line $_"; $x=$phys{$A[0]}; 
    defined $x||die "Not defined for $A[0]"; print "$x\n"; } } ' \
 $mlist > test/phys.seq


cat test/phys.seq | perl -e '
  open(F, "$ARGV[0]") || die "Opening MMF $ARGV[0]";
  while(<F>){
    if(m/\~h \"(\S+)\"/) {
      $phys_name = $1;
      $state_seq = "";
      while(<F>) { 
        if(m/ENDHMM/) { last; }
        elsif(m/\~s \"(\S+)\"/) {
          $state_seq = $state_seq . $1 . "\n";
        }
      }
      $m2s{$phys_name} = $state_seq;
    }
  }
   while(<STDIN>) {
       @A=split(" ",$_);
       if(@A==1) { 
          $x=$m2s{$A[0]};  
          defined $x||die "No phys model $A[0]\n";
          print $x;
       }
    }  '   $mmf > test/htk.state.seq


# transducer from pdf+1 to tid.
../../../src/bin/make-pdf-to-tid-transducer --print-args=false kaldi.mdl > test/p2t.fst

# Make symbol table like statesyms.txt but with offset of 1.
cat statesyms.txt | awk 'BEGIN{ print "<eps> 0"; } {printf("%s %d\n", $1, $2+1);} ' > test/pdfplusone.txt

fstcompose ilabel_map.fst test/fst.test | fstcompose Ha.fst -  | fstrandgen | fstrmsymbols disambig_tstate.list | fstcompose test/p2t.fst - | fstprint --isymbols=test/pdfplusone.txt --osymbols=words.txt | awk '{print $3}' | grep -v -w eps |  uniq | awk '{if(NF>0){print;}}' > test/kaldi.state.seq


# Remove the individual silence states and replace with one, because the silence model
# has backwards transitions which can cause fstrandgen to produce paths that differ
# from the HTK ones.

cat test/kaldi.state.seq | sed s:silsp_2:silsp: |  sed s:silsp_3:silsp: |  sed s:silsp_4:silsp:  \
  | uniq > test/kaldi.state.seq.onesil

cat test/htk.state.seq | sed s:silsp_2:silsp: |  sed s:silsp_3:silsp: |  sed s:silsp_4:silsp:  \
  | uniq > test/htk.state.seq.onesil

cmp test/kaldi.state.seq.onesil  test/htk.state.seq.onesil  || exit 1;

echo "Test succeeded."

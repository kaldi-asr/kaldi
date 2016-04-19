#!/bin/bash 

# begin options
exclude_uttlist=
subset_time_ratio=0.1
random=false
data2=
# end options
function print_options {
  echo 
  echo "Select subset data from the source data, data2 is optional, and"
  echo "it is complementary with the data, taking sdata as the super set."
  echo
  echo "Usage: $0 [options] <sdata> <data>"
cat<<END

 [options]:
 --exclude-uttlist			# value, "$exclude_uttlist"
 --subset_time_ratio			# value, $subset_time_ratio
 --random				# value, $random
 --data2				# value, "$data2"
END
}

. utils/parse_options.sh || \
{ echo "$0: ERROR, utils/parse_options.sh expected"; exit 1; }

if [ $# -lt 2 ]; then
  print_options
  exit 1
fi

sdata=$1
data=$2

for x in $sdata/{segments,utt2spk}; do
  [ -f $x ] || { echo "$0: ERROR, file $x expected"; exit 1; }
done
[ -d $data ] || mkdir -p $data
if [ ! -z $exclude_uttlist ]; then
  cat $sdata/segments | \
  perl -e '($uttlist) = @ARGV;
    open(U, "$uttlist") or die;
    while(<U>) {
      chomp; @A = split(" "); $uttName = $A[0];
      $vocab{$uttName} ++;
    }  close U;
    while(<STDIN>) {
      chomp;
      @A = split(" "); $uttName = $A[0];
      if(not exists $vocab{$uttName}) {
        print $_."\n";
      }
    }
  '  $exclude_uttlist  > $data/.uttlist
  numLine=$(wc -l < $data/.uttlist)
  if [ $numLine -eq 0 ]; then
    echo "ERROR, zero utterance extracted" && exit 1
  fi
  echo "$numLine utterances are extracted"
  utils/subset_data_dir.sh --utt-list $data/.uttlist  $sdata $data
  echo "Done !" && exit 0
fi
[ ! -z $data2 ] &&  mkdir -p $data2

x=$subset_time_ratio
if [ $(bc -l <<< "$x < 1 && $x > 0") -ne 1 ]; then
  echo "$0: ERROR, subset_time_ratio($subset_time_ratio) should be (0, 1)" && exit 1
fi

tot_hour=$(cat $sdata/segments| awk '{x+=$4-$3;}END{print x/3600;}')
sel_hour=$(perl -e "print $tot_hour*$subset_time_ratio")
echo "$0: tot_hour=$tot_hour, sel_hour=$sel_hour"
super_utt2spk=$sdata/utt2spk
if $random; then
  super_utt2spk_new=$data/.feats_shuffled.scp
  cat $super_utt2spk | \
  perl -MList::Util -e 'print List::Util::shuffle <>' > $super_utt2spk_new
  line_num_new=$(wc -l < $super_utt2spk_new)
  [ -z $line_num_new ] && \
  { echo "$0: ERROR, file $super_utt2spk_new is failed to generate"; exit 1; }
  line_num=$(wc -l < $super_utt2spk)
  [ $line_num_new -eq $line_num ] || \
  { echo "$0: ERROR, original lines: $line_num ($super_utt2spk), new lines: $line_num_new($super_utt2spk_new), are mismatched"; exit 1; }
  super_utt2spk=$super_utt2spk_new
fi
uttlist2=/dev/null
[ ! -z $data2 ] && uttlist2=$data2/uttlist
cat $super_utt2spk | \
source/egs/swahili/subset_lines.pl $sel_hour $sdata/segments $data/uttlist \
>$uttlist2
utils/subset_data_dir.sh --utt-list $data/uttlist  $sdata $data
[ ! -z $data2 ] && \
utils/subset_data_dir.sh --utt-list $data2/uttlist $sdata  $data2

echo "$0: Done !"



#! /usr/bin/perl

#*****************************************************************************
# IrstLM: IRST Language Model Toolkit
# Copyright (C) 2007 Marcello Federico, ITC-irst Trento, Italy

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

#******************************************************************************



#first pass: read dictionary and generate 1-grams
#second pass: 
#for n=2 to N
#  foreach n-1-grams
#      foreach  n-grams with history n-1
#          compute smoothing statistics
#          store successors
#      compute back-off probability
#      compute smoothing probability
#      write n-1 gram with back-off prob 
#      write all n-grams with smoothed probability

use strict;
use Getopt::Long "GetOptions";
use File::Basename;

my $gzip=`which gzip 2> /dev/null`;
my $gunzip=`which gunzip 2> /dev/null`;
chomp($gzip);
chomp($gunzip);
my $cutoffword="<CUTOFF>"; #special word for Google 1T-ngram cut-offs 
my $cutoffvalue=39;   #cut-off threshold for Google 1T-ngram cut-offs 

#set defaults for optional parameters
my ($verbose,$size,$ngrams,$sublm)=(0, 0, undef, undef);
my ($witten_bell,$good_turing,$shift_beta,$improved_shift_beta,$stupid_backoff)=(0, 0, "", "", "");
my ($witten_bell_flag,$good_turing_flag,$shift_beta_flag,$improved_shift_beta_flag,$stupid_backoff_flag)=(0, 0, 0, 0, 0);
my ($freqshift,$prune_singletons,$prune_thr_str,$cross_sentence)=(0, 0, "", 0);

my $help = 0;
$help = 1 unless
&GetOptions('size=i' => \$size,
'freq-shift=i' => \$freqshift, 
'ngrams=s' => \$ngrams,
'sublm=s' => \$sublm,
'witten-bell' => \$witten_bell,
'good-turing' => \$good_turing,
'shift-beta=s' => \$shift_beta,
'improved-shift-beta=s' => \$improved_shift_beta,
'stupid-backoff' => \$stupid_backoff,
'prune-singletons' => \$prune_singletons,
'pft|PruneFrequencyThreshold=s' => \$prune_thr_str,
'cross-sentence' => \$cross_sentence,
'h|help' => \$help,
'verbose' => \$verbose);


if ($help || !$size || !$ngrams || !$sublm) {
	my $cmnd = basename($0);
  print "\n$cmnd - estimates single LMs\n",
	"\nUSAGE:\n",
	"       $cmnd [options]\n",
	"\nOPTIONS:\n",
	"       --size <int>          maximum n-gram size for the language model\n",
	"       --ngrams <string>     input file or command to read the ngram table\n",
	"       --sublm <string>      output file prefix to write the sublm statistics \n",
	"       --freq-shift <int>    (optional) value to be subtracted from all frequencies\n",
	"       --witten-bell         (optional) use Witten-Bell linear smoothing (default) \n",
	"       --shift-beta <string> (optional) use Shift-Beta smoothing with statistics in <string>\n",
	"       --improved-shift-beta <string> (optional) use Improved Shift-Beta smoothing with statistics in <string>, similar to Improved Kneser Ney but without corrected counts\n",
	"       --good-turing         (optional) use Good-Turing linear smoothing\n",
	"       --stupid-backoff      (optional) use Stupid-Backoff smoothing\n",
	"       --prune-singletons    (optional) remove n-grams occurring once, for n=3,4,5,... (disabled by default)\n",
	"       -pft, --PruneFrequencyThreshold <string>	(optional) pruning frequency threshold for each level; comma-separated list of values; (default is \"0,0,...,0\", for all levels)\n",
	"       --cross-sentence      (optional) include cross-sentence bounds (disabled by default)\n",
	"       --verbose             (optional) print debugging info\n",
	"       -h, --help            (optional) print these instructions\n",
	"\n";
	
  exit(1);
}

$good_turing_flag = 1 if ($good_turing);
die "build-sublm: This LM is no more supported\n\n" if ($good_turing_flag==1);

$witten_bell_flag = 1 if ($witten_bell);
$shift_beta_flag = 1 if ($shift_beta);
$stupid_backoff_flag = 1 if ($stupid_backoff);
$improved_shift_beta_flag = 1 if ($improved_shift_beta);
$witten_bell = $witten_bell_flag = 1 if ($witten_bell_flag + $shift_beta_flag + $improved_shift_beta_flag + $stupid_backoff_flag) == 0;

print STDERR  "build-sublm: size=$size ngrams=$ngrams sublm=$sublm witten-bell=$witten_bell shift-beta=$shift_beta improved-shift-beta=$improved_shift_beta stupid-backoff=$stupid_backoff prune-singletons=$prune_singletons cross-sentence=$cross_sentence PruneFrequencyThreshold=$prune_thr_str\n" if $verbose;


die "build-sublm: choose only one smoothing method\n" if ($witten_bell_flag + $shift_beta_flag + $improved_shift_beta_flag + $stupid_backoff_flag) > 1;

die "build-sublm: value of --size must be larger than 0\n" if $size<1;



my @pruneFreqThr=();
my $i=0;
while ($i<=$size){
	$pruneFreqThr[$i++]=0;
}

print STDERR "Pruning frequency threshold values:$prune_thr_str\n" if ($verbose);

my @v=split(/,/,$prune_thr_str);
$i=0;
while ($i<scalar(@v)){
	$pruneFreqThr[$i+1]=$v[$i];
	$i++;
	if ($i>=$size){
		print STDERR "too many pruning frequency threshold values; kept the first values and skipped the others\n" if ($verbose);
		last;	
	};
}

$i=1;
while ($i<=$size){
	if ($pruneFreqThr[$i] < $pruneFreqThr[$i-1]){
		$pruneFreqThr[$i]=$pruneFreqThr[$i-1];
		print STDERR "the value of the pruning frequency threshold for level $i has been adjusted to value $pruneFreqThr[$i]\n" if ($verbose);
	}
	$i++;
}

if ($verbose){
	$i=0;
	while ($i<=$size){
		print STDERR "pruneFreqThr[$i]=$pruneFreqThr[$i]\n";
		$i++;
	}
}

my $log10=log(10.0);	   #service variable to convert log into log10
my $oldwrd="";		   #variable to check if 1-gram changed 
my @cnt=();		   #counter of n-grams
my $totcnt=0;		   #total counter of n-grams
my ($ng,@ng);		   #read ngrams
my $ngcnt=0;		   #store ngram frequency
my $n;

print STDERR  "Collecting 1-gram counts\n" if $verbose;

open(INP,"$ngrams") || open(INP,"$ngrams|")  || die "cannot open $ngrams\n";
open(GR,"|$gzip -c >${sublm}.1gr.gz") || die "cannot create ${sublm}.1gr.gz\n";

while ($ng=<INP>) {
  
  chomp($ng);  @ng=split(/[ \t]+/,$ng);  $ngcnt=(pop @ng) - $freqshift;
  
	#	warn "ng: |@ng| ngcnt:$ngcnt\n";
	
  if ($oldwrd ne $ng[0]) {
		#    warn "$totcnt,$oldwrd,$ng[0]\n" if $oldwrd ne '';
    printf (GR "%s\t%s\n",$totcnt,$oldwrd) if $oldwrd ne '';
    $totcnt=0;$oldwrd=$ng[0];
  }
  
  #update counter
  $totcnt+=$ngcnt;
}

printf GR "%s\t%s\n",$totcnt,$oldwrd;
close(INP);
close(GR);

my (@h,$h,$hpr);	      #n-gram history 
my (@dict,$code);	      #sorted dictionary of history successors
my ($diff,$singlediff,$diff1,$diff2,$diff3); #different successors of history
my (@n1,@n2,@n3,@n4,@uno3);  #IKN: n-grams occurring once or twice ...
my (@beta,$beta);	     #IKN: n-grams occurring once or twice ...
my $locfreq;

#collect global statistics for (Improved) Shift-Beta smoothing
if ($shift_beta_flag || $improved_shift_beta_flag) {
  my $statfile=$shift_beta || $improved_shift_beta;
  print STDERR  "load \& merge IKN statistics from $statfile \n" if $verbose;
  open(IKN,"$statfile") || open(IKN,"$statfile|")  || die "cannot open $statfile\n";
  while (<IKN>) {
    my($lev,$n1,$n2,$n3,$n4,$uno3)=$_=~/level: (\d+)  n1: (\d+) n2: (\d+) n3: (\d+) n4: (\d+) unover3: (\d+)/;
    $n1[$lev]+=$n1;$n2[$lev]+=$n2;$n3[$lev]+=$n3;$n4[$lev]+=$n4;$uno3[$lev]+=$uno3;
		print STDERR  "from $statfile level $lev: n1:$n1 n2:$n2 n3:$n3 n4:$n4 uno3:$uno3\n";
		print STDERR  "level $lev: n1[$lev]:$n1[$lev] n3[$lev]:$n2[$lev]  n3[$lev]:$n3[$lev] n4[$lev]:$n4[$lev] uno3[$lev]:$uno3[$lev]\n";
  }
	if ($verbose){
		for (my $lev=1;$lev<=$#n1;$lev++) {
			print STDERR  "level $lev: n1[$lev]:$n1[$lev] n3[$lev]:$n2[$lev]  n3[$lev]:$n3[$lev] n4[$lev]:$n4[$lev] uno3[$lev]:$uno3[$lev]\n";
		}
	}
  close(IKN);
}

print STDERR  "Computing n-gram probabilities:\n" if $verbose;

foreach ($n=2;$n<=$size;$n++) {
	
  $code=-1;@cnt=(); @dict=(); $totcnt=0;$diff=0; $singlediff=1; $diff1=0; $diff2=0; $diff3=0; $oldwrd=""; 
	
  #compute smothing statistics         
  my (@beta,$beta);               
	
  if ($stupid_backoff_flag) {
		$beta=0.4;
		print STDERR  "Stupid-Backoff smoothing: beta $n: $beta\n" if $verbose;
	}
	
  if ($shift_beta_flag) {
    if ($n1[$n]==0 || $n2[$n]==0) {
      print STDERR  "Error in Shift-Beta smoothing statistics: resorting to Witten-Bell\n" if $verbose;  
      $beta=0;  
    } else {
      $beta=$n1[$n]/($n1[$n] + 2 * $n2[$n]); 
      print STDERR  "Shift-Beta smoothing: beta $n: $beta\n" if $verbose;  
    }
  }
	
  if ($improved_shift_beta_flag) {
		
    my $Y=$n1[$n]/($n1[$n] + 2 * $n2[$n]);
		
    if ($n3[$n] == 0 || $n4[$n] == 0 || $n2[$n] <= $n3[$n] || $n3[$n] <= $n4[$n]) {
      print STDERR  "Warning: higher order count-of-counts are wrong\n" if $verbose;
      print STDERR  "Fixing this problem by resorting only on the lower order count-of-counts\n" if $verbose;     
      $beta[1] = $Y;
      $beta[2] = $Y;
      $beta[3] = $Y;
    } else {
      $beta[1] = 1 - 2 * $Y * $n2[$n] / $n1[$n];
      $beta[2] = 2 - 3 * $Y * $n3[$n] / $n2[$n];
      $beta[3] = 3 - 4 * $Y * $n4[$n] / $n3[$n];
    }
		print STDERR  "Improved-Shift-Beta  smoothing: level:$n beta[1]:$beta[1] beta[2]:$beta[2] beta[3]:$beta[3]\n" if $verbose; 
  }
	
  open(HGR,"$gunzip -c ${sublm}.".($n-1)."gr.gz |") || die "cannot open ${sublm}.".($n-1)."gr.gz\n";
  open(INP,"$ngrams") || open(INP,"$ngrams |")  || die "cannot open $ngrams\n";
  open(GR,"| $gzip -c >${sublm}.${n}gr.gz");
  open(NHGR,"| $gzip -c > ${sublm}.".($n-1)."ngr.gz") || die "cannot open ${sublm}.".($n-1)."ngr.gz";
	
  my $ngram;
  my ($reduced_h, $reduced_ng) = ("", "");
	
  $ng=<INP>; chomp($ng); @ng=split(/[ \t]+/,$ng); $ngcnt=(pop @ng) - $freqshift;
  $h=<HGR>; chomp($h); @h=split(/[ \t]+/,$h); $hpr=shift @h;
  $reduced_ng=join(" ",@ng[0..$n-2]);
  $reduced_h=join(" ",@h[0..$n-2]);
	
  @cnt=(); @dict=();
  $code=-1; $totcnt=0; $diff=0; $singlediff=0; $diff1=0; $diff2=0; $diff3=0; $oldwrd="";
  do{
		
    #load all n-grams starting with history h, and collect useful statistics 
		
    while ($reduced_h eq $reduced_ng){ #must be true the first time!

      if ($oldwrd ne $ng[$n-1]) { #could this be otherwise? [Marcello 22/5/09]
				$oldwrd=$ng[$n-1];
				++$code;
			}
			
			$dict[$code]=$ng[$n-1];
      $cnt[$code]+=$ngcnt;
			$totcnt+=$ngcnt;
			
      $ng=<INP>;

			if (defined($ng)){
				chomp($ng);
				@ng=split(/[ \t]+/,$ng);$ngcnt=(pop @ng) - $freqshift;  
				$reduced_ng=join(" ",@ng[0..$n-2]);
			}
			else{
				last;
			}
    }	
		
		$diff=scalar(@cnt);	
		for (my $c=0;$c<scalar(@cnt);++$c){
			$singlediff++ if $cnt[$c]==1;
			
      if ($diff>1 && $dict[$c] eq $cutoffword) { # in google n-grams
				#find estimates for remaining diff and singlediff
				#proportional estimate
				$diff--;		#remove cutoffword
				my $concentration=1.0-($diff-1)/$totcnt;
				my $mass=1;		#$totcnt/($totcnt+$ngcnt);
				my $index=(1-($concentration * $mass))/(1-1/$cutoffvalue) + (1/$cutoffvalue);
				my $cutoffdiff=int($ngcnt * $index);
				$cutoffdiff=1 if $cutoffdiff==0;
				print STDERR "diff $diff $totcnt cutofffreq $ngcnt -- cutoffdiff: $cutoffdiff\n";
				print STDERR "concentration:",$concentration," mass:", $mass,"\n";
				$diff+=$cutoffdiff;
      }
		}

		
    if ($improved_shift_beta) { 
      for (my $c=0;$c<=$code;$c++) {
				$diff1++ if $cnt[$c]==1;
				$diff2++ if $cnt[$c]==2;
				$diff3++ if $cnt[$c]>=3;
      }
    }
		
    #print smoothed probabilities
    my $boprob=0;		#accumulate pruned probabilities 
    my $prob=0;
		my $boprob_correction=0; #prob for the correction due to singleton pruning
		
		if ($totcnt>0){	
			for (my $c=0;$c<=$code;$c++) {
				
				$ngram=join(" ",$reduced_h,$dict[$c]);

				print STDERR "totcnt:$totcnt diff:$diff singlediff:$singlediff\n" if $totcnt+$diff+$singlediff==0;
				
				if ($shift_beta && $beta>0) {
					$prob=($cnt[$c]-$beta)/$totcnt;
				} elsif ($improved_shift_beta) {
					my $b=($cnt[$c]>= 3? $beta[3]:$beta[$cnt[$c]]);
					$prob=($cnt[$c] - $b)/$totcnt;
				} elsif ($stupid_backoff) {
					$prob=$cnt[$c]/$totcnt;
				} else { ### other smoothing types, like Witten-Bell
					$prob=$cnt[$c]/($totcnt+$diff);
				}
				
				## skip n-grams containing OOV
				##		  if (&containsOOV($ngram)){ print STDERR "ngram:|$ngram| contains OOV --> hence skip\n";  next; }
				
				## skip also n-grams containing eos symbols not at the final
				##			if (&CrossSentence($ngram)){ print STDERR "ngram:|$ngram| is Cross Sentence --> hence skip\n";  next; }
				
				
				#rm singleton n-grams for (n>=3), if flag is active
				#rm n-grams (n>=2) containing cross-sentence boundaries, if flag is not active
				#rm n-grams containing <unk> or <cutoff> except for 1-grams
				
				#warn "considering $size $n |$ngram|\n";				
				if (($prune_singletons && $n>=3 && $cnt[$c]==1) ||
					(!$cross_sentence && &CrossSentence($ngram)) || 
					(&containsOOV($dict[$c])) ||
					($n>=2 && &containsOOV($h)) ||	
					($dict[$c] eq $cutoffword) 
					)
				{						
					$boprob+=$prob;
					
					if ($n<$size) {	#output this anyway because it will be an history for n+1 
						printf GR "%f\t%s %s\n",-10000,$reduced_h,$dict[$c];
					}
				} else {
					if ($cnt[$c] > $pruneFreqThr[$n]){
						# print unpruned n-1 gram
						my $logp=log($prob)/$log10;
						printf(GR "%f\t%s %s\n",($logp>0?0:$logp),$reduced_h,$dict[$c]);
					}else{
						if ($n<$size) {	#output this anyway because it will be an history for n+1 
							printf GR "%f\t%s %s\n",-10000,$reduced_h,$dict[$c];
						}
					}
				}
			}
		}else{
			$boprob=0;
		}
		
		if (($prune_singletons && $n>=3)){
			if ($shift_beta && $beta>0) { # correction due to singleton pruning
				$boprob_correction += (1.0-$beta) * $singlediff / $totcnt;
			} elsif ($improved_shift_beta) { # correction due to singleton pruning
				$boprob_correction += (1-$beta[1]) * $singlediff / $totcnt;
			} elsif ($stupid_backoff) { # correction due to singleton pruning
				$boprob_correction += $singlediff/($totcnt);
			} else { # correction due to singleton pruning
				$boprob_correction += $singlediff/($totcnt+$diff);
			} 
		}
		else{
			$boprob_correction = 0;
		}

		$boprob=$boprob_correction;
			
    #rewrite history including back-off weight
		
    #check if history has to be pruned out
    if ($hpr==-10000) {
      #skip this history
    } elsif ($shift_beta && $beta>0) {
			print STDERR "wrong division: considering rewriting history --- h:|$h| --- hpr=$hpr --- totcnt:$totcnt -- denumerator:",($totcnt),"\n" if $totcnt==0;
      my $lambda=$beta * $diff/$totcnt; 	
      my $logp=log($boprob+$lambda)/$log10;
      printf NHGR "%s\t%f\n",$h,($logp>0?0:$logp);
    } elsif ($improved_shift_beta) {
			print STDERR "wrong division: considering rewriting history --- h:|$h| --- hpr=$hpr --- totcnt:$totcnt -- denumerator:",($totcnt),"\n" if $totcnt==0;
      my $lambda=($beta[1] * $diff1 + $beta[2] * $diff2 + $beta[3] * $diff3)/$totcnt; 	  
      my $logp=log($boprob+$lambda)/$log10;
      printf NHGR "%s\t%f\n",$h,($logp>0?0:$logp);
    } elsif ($stupid_backoff) {
			print STDERR "wrong division: considering rewriting history --- h:|$h| --- hpr=$hpr --- totcnt:$totcnt -- denumerator:",($totcnt),"\n" if $totcnt==0;
      my $lambda=$beta;
			my $logp=log($lambda)/$log10;
      printf NHGR "%s\t%f\n",$h,($logp>0?0:$logp);
    } else {
			print STDERR "wrong division: considering rewriting history --- h:|$h| --- hpr=$hpr --- totcnt:$totcnt diff:$diff -- denumerator:",($totcnt+$diff),"\n" if $totcnt+$diff==0;
      my $lambda=$diff/($totcnt+$diff); 
      my $logp=log($boprob+$lambda)/$log10;
      printf NHGR "%s\t%f\n",$h,($logp>0?0:$logp);
    }     
		
    #reset smoothing statistics
    $code=-1;@cnt=(); @dict=(); $totcnt=0;$diff=0;$singlediff=0;$oldwrd="";$diff1=0;$diff2=0;$diff3=0;$locfreq=0;
		
    #read next history
    $h=<HGR>;
		
    if (defined($h)){
      chomp($h); @h=split(/[ \t]+/,$h); $hpr=shift @h;
      $reduced_h=join(" ",@h[0..$n-2]);
    }else{
      die "ERROR: Something could be wrong: history are terminated before ngrams!" if defined($ng);
    }
  }until (!defined($ng));		#n-grams are over
	
  close(HGR); close(INP); close(GR); close(NHGR);
	
  rename("${sublm}.".($n-1)."ngr.gz","${sublm}.".($n-1)."gr.gz");
}   


#check if n-gram contains cross-sentence boundaries
sub CrossSentence(){
  my ($ngram) = @_;
  if ($ngram=~/<\/s> /i) { #if </s> occurs not only in the last place
		print STDERR  "check CrossSentence ngram:|$ngram| is CrossSentence\n" if $verbose;
    return 1;
  }
  return 0;
}

#check if n-gram contains OOV
sub containsOOV(){
  my ($ngram) = @_;
  if ($ngram=~/<UNK>/i){
		print STDERR  "check containsOOV ngram:|$ngram| contains OOV\n" if $verbose;
    return 1;
  }
  return 0;
}

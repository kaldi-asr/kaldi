#!/usr/bin/env perl
#Argument -> enhancement_method enhancement_directory destination_directory chime_RIR_directory

my $enhancement_method = $ARGV[0];
my $enhancement_directory = $ARGV[1];
my $chime_RIR_directory = $ARGV[3];
my $destination_directory = $ARGV[2]; 
`rm PESQ_log`;

my @et05_files = `ls $enhancement_directory/et05_*_simu/*.wav`;
$KALDI_ROOT = $ENV{'KALDI_ROOT'};
my $nFiles = 0;
$tMOS = 0;
foreach my $line (@et05_files) {
chomp($line);
$nFiles = $nFiles + 1;
my $target_filename = substr($line, rindex($line,"/")+1, length($line)-rindex($line,"/")-1);
my @fields = split /_/, $target_filename;
my @fields2 = split /\./, $fields[2];
my $noise_cap =  $fields2[0];
my $noise=lc $fields2[0];
print `PESQ +16000 $chime_RIR_directory/et05_${noise}_simu/$fields[0]_$fields[1]_${noise_cap}.CH5.Clean.wav $line > temp`;
my $pesq_score=`tail -1 temp | cut -d " " -f8 | cut -d\$'\t' -f2`;
$tMOS=$tMOS+$pesq_score;
`cat temp >> PESQ_log`;
}
open FH, ">", "${destination_directory}/${enhancement_method}_et05_PESQ" or die $!;
my $avg_mos=$tMOS/$nFiles;
print FH $avg_mos;
close FH;

my @dt05_files = `ls $enhancement_directory/dt05_*_simu/*.wav`;
$KALDI_ROOT = $ENV{'KALDI_ROOT'};
my $nFiles = 0;
$tMOS = 0;
foreach my $line (@dt05_files) {
chomp($line);
$nFiles = $nFiles + 1;
my $target_filename = substr($line, rindex($line,"/")+1, length($line)-rindex($line,"/")-1);
my @fields = split /_/, $target_filename;
my @fields2 = split /\./, $fields[2];
my $noise_cap =  $fields2[0];
my $noise=lc $fields2[0];
print `PESQ +16000 $chime_RIR_directory/dt05_${noise}_simu/$fields[0]_$fields[1]_${noise_cap}.CH5.Clean.wav $line > temp`;
my $pesq_score=`tail -1 temp | cut -d " " -f8 | cut -d\$'\t' -f2`;
$tMOS=$tMOS+$pesq_score;
`cat temp >> PESQ_log`;
}
open FH, ">", "${destination_directory}/${enhancement_method}_dt05_PESQ" or die $!;
my $avg_mos=$tMOS/$nFiles;
print FH $avg_mos;
close FH;

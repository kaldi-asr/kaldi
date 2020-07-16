#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University   
# Apache 2.0.

stage=0

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ] && . ./cmd.sh
[ -f /export/babel/data/software/env.sh ] && . /export/babel/data/software/env.sh

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: create_shadow_dataset.sh <dest-data-dir> <src-data-dir1> <src-data-dir2> "
  exit 1
fi

dest=$1
src1=$2
src2=$3

mkdir -p $dest/kws

if [ $stage -le 0 ] ; then
  utils/combine_data.sh $dest $src1 $src2 || exit 1
fi

if [ $stage -le 1 ] ; then
  #zkombinovat ecf
  echo "Combining ECF files..."
  perl -e '
    #binmode STDIN, ":utf8"; 
    binmode STDOUT, ":utf8"; 

    use XML::Simple;
    use Data::Dumper;

    use strict;
    use warnings;


    my $src1 = XMLin($ARGV[0]);
    my $src2 = XMLin($ARGV[1]);
    my $tgt={};
    my %filename_hash;

    my $expected_duration=0.0;
    my $duration=0.0;

    if ( $src1->{language} ne $src2->{language} ) {
        die "ECF languages differ in the source ecf.xml files"
    }
    $expected_duration=$src1->{source_signal_duration} + $src2->{source_signal_duration};

    $tgt->{source_signal_duration} = $expected_duration;
    $tgt->{language}=$src1->{language};
    $tgt->{version}="Generated automatically by the shadow_set.sh script";
    $tgt->{excerpt}= [];

    #print Dumper(\$src1);
    foreach my $excerpt ( @{$src1->{excerpt}} ) {
       push @{$tgt->{excerpt}}, $excerpt;
       if ( exists $filename_hash{$excerpt->{audio_filename}} ) {
          print STDERR "[WARN]: Duplicate filename $excerpt->{audio_filename} \n"
       } else {
         $duration += $excerpt->{dur} ;
         $filename_hash{$excerpt->{audio_filename}} = $excerpt;
      }
    }
    foreach my $excerpt ( @{$src2->{excerpt}} ) {
       push @{$tgt->{excerpt}}, $excerpt;
       if ( exists $filename_hash{$excerpt->{audio_filename}} ) {
          print STDERR "[WARN]: Duplicate filename $excerpt->{audio_filename} \n"
       } else {
         $duration += $excerpt->{dur} ;
         $filename_hash{$excerpt->{audio_filename}} = $excerpt;
      }
    }
    $tgt->{source_signal_duration} = $duration;

    my $tgtxml = XMLout($tgt, RootName=>"ecf");
    print $tgtxml;
  ' $src1/kws/ecf.xml $src2/kws/ecf.xml > $dest/kws/ecf.xml
fi

if [ $stage -le 2 ] ; then
  #zkombinovat kwlist
  echo "Combining the KWLIST files"
  perl -e '
    #binmode STDIN, ":utf8"; 
    binmode STDOUT, ":utf8"; 

    use XML::Simple;
    use Data::Dumper;

    use strict;
    use warnings;

    my $src1 = XMLin($ARGV[0],  ForceArray => 1);
    my $src2 = XMLin($ARGV[1],  ForceArray => 1);
    my $tgt={};
    my %kwid_hash;

    if ( $src1->{compareNormalize} ne $src2->{compareNormalize} ) {
        die "KWLIST compareNormalize attributes differ in the source kwlist.xml files";
    }
    if ( $src1->{language} ne $src2->{language} ) {
        die "KWLIST languages differ in the source kwlist.xml files";
    }
    
    $tgt->{ecf_filename} = "";
    $tgt->{language}=$src1->{language};
    $tgt->{compareNormalize}=$src1->{compareNormalize};
    $tgt->{encoding}=$src1->{encoding};
    $tgt->{version}="1";
    $tgt->{kw}= [];


    foreach my $kw ( @{$src1->{kw}} ) {
       $kw->{kwid} = $kw->{kwid} . "-A";
       if ( exists $kwid_hash{$kw->{kwid}} ) {
          print STDERR "[WARN]: Duplicate kwid $kw->{kwid}\n";
       } else {
         $kwid_hash{$kw->{kwid}} = $kw;
       }
       push @{$tgt->{kw}}, $kw;
    }
    foreach my $kw ( @{$src2->{kw}} ) {
       $kw->{kwid} = $kw->{kwid} . "-B";
       if ( exists $kwid_hash{$kw->{kwid}} ) {
          print STDERR "[WARN]: Duplicate kwid $kw->{kwid}\n";
       } else {
         $kwid_hash{$kw->{kwid}} = $kw;
       }
       push @{$tgt->{kw}}, $kw;
    }

    my $tgtxml = XMLout($tgt, RootName=>"kwlist", KeyAttr=>"");
    print $tgtxml;
  ' $src1/kws/kwlist.xml $src2/kws/kwlist.xml > $dest/kws/kwlist.xml || exit 1
fi

if [ $stage -le 3 ] ; then
  echo "Making KWLIST maps"
  perl -e '
    #binmode STDIN, ":utf8"; 
    binmode STDOUT, ":utf8"; 

    use XML::Simple;
    use Data::Dumper;

    use strict;
    use warnings;

    my $src1 = XMLin($ARGV[0],  ForceArray => 1);
    open TGT_DEV, ">", $ARGV[1] or die $!;
    open TGT_TST, ">", $ARGV[2] or die $!;

    foreach my $kw ( @{$src1->{kw}} ) {
        if ( $kw->{kwid} =~ "KW.+-A\$" ) {
            my $new_kw = $kw->{kwid};
            my $old_kw = substr $new_kw, 0, -2;
            print TGT_DEV "$old_kw\t$new_kw\n";
        } elsif ( $kw->{kwid} =~ "KW.+-B\$" ) {
            my $new_kw = $kw->{kwid};
            my $old_kw = substr $new_kw, 0, -2;
            print TGT_TST  "$old_kw\t$new_kw\n";
        } else {
            die "Unsupported or unknown KW ID: $kw->{kwid}\n";
        }
    }
  ' $dest/kws/kwlist.xml $dest/kws/kws_map.dev.txt $dest/kws/kws_map.test.txt || exit 1
fi

#RTTM file is not necessary

utils/fix_data_dir.sh data/shadow.uem

exit 0


#!/usr/bin/env perl
#===============================================================================
# Copyright 2017  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

use strict;
use warnings;
use utf8;


sub read_whole_file {
  my $file = shift;
  local $/=undef;
  open(my $in, $file) or die "Cannot open $file";
  binmode $in;
  $file = <$in>;
  $file =~ s/#.*[\$\n]//g;
  $file =~ s/\\\n/ /g; 
  close($in);
  return $file;
}

sub extract_statement {
  my $statement = shift;
  my $file = shift;
  
  my $content = undef;
  if ($file =~ /[\n^]\Q$statement\E *= *([^\n]+)[\n\$]/) {
    $content = $1;
    #print "$statement = \"$content\"\n";
    $content = join(" ", split(" ", $content));
    if (!$content) {
        $content = " ";
    }
  } 
  return $content;
}


sub write_root_cmake {
  my $kaldimk = shift;
  my $directories = shift;
  my $out = shift;
  
  #print $kaldimk;
  
  my $openfstinc = extract_statement("OPENFSTINC", $kaldimk) or die "No OPENFSTINC";
  my $openfstlib = extract_statement("OPENFSTLIBS", $kaldimk) or die "No OPENFSTLIBS";
  my $openfstflags = extract_statement("OPENFSTLDFLAGS", $kaldimk) or die "No OPENFSTLDFLAGS";

  my $cxxflags = extract_statement("CXXFLAGS", $kaldimk) or die "No CXXFLAGS!";
  my $ldflags = extract_statement("LDFLAGS", $kaldimk) or die "No LDFLAGS!";
  my $ldlibs = extract_statement("LDLIBS", $kaldimk) or die "No LDLIBS!";

  my @cxxflags_array = split / /, $cxxflags;
  my @defines_from_flags;
  my @includes_from_flags;
  my $have_mkl;
  foreach my $param (@cxxflags_array) {
    if ($param =~ /-D/) {
      $param =~ s/-D//;
      if ($param eq "HAVE_MKL") {
        $have_mkl = 1;
      }
      push @defines_from_flags, $param;       
    }
    elsif ($param =~ /-I/) {
      $param =~ s/-I//;
      push @includes_from_flags, $param;       
    }
  }
  
  $cxxflags =~ s/-D[^ ]+//g;
  $cxxflags =~ s/-I[^ ]+//g;
  $cxxflags =~ s/\$[^\)]+\)//g;
  $cxxflags =~ s/ *$//g;
  $cxxflags =~ s/^ *//g;
  $ldlibs =~ s/\$[^\)]+\)//g;
  $ldlibs =~ s/^ *//g;
  $ldlibs =~ s/ *$//g;
  $ldflags =~ s/\$[^\)]+\)//g;
  $ldflags =~ s/^ *//g;
  $ldflags =~ s/ *$//g;
  #print $cxxflags . "\n";
  #print $ldflags . "\n";
  #print $ldlibs . "\n";



  open(my $f, ">", "$out/CMakeLists.txt");
  print $f "cmake_minimum_required (VERSION 2.8.11)\n";
  print $f "project(kaldi)\n";
  print $f "enable_testing()\n";
  print $f "include_directories(\${CMAKE_SOURCE_DIR})\n";
  print $f "include_directories($openfstinc)\n";
  foreach my $param (@includes_from_flags) {
    print $f "include_directories($param)\n";
  }
  if ($have_mkl) {
    my $mkl = extract_statement("MKLDIR", $kaldimk) or die "No MKLDIR!";
    print $f "include_directories(\"$mkl" . '\\\\' . "include\")\n";
  }
  #print $f "add_definitions(-DHAVE_CLAPACK)\n";
  print $f "add_definitions(-DHAVE_EXECINFO=1)\n";
  print $f "add_definitions(-DHAVE_CXXABI_H)\n";
  print $f "add_definitions(-DKALDI_DOUBLEPRECISION=0)\n";
  foreach my $param (@defines_from_flags) {
    print $f "add_definitions(-D$param)\n";
  }
  print $f "option (BUILD_SHARED_LIBS \"Build shared libraries.\" \"SHARED\")\n";
  print $f "option (BUILD_USE_SOLUTION_FOLDERS \"Enable grouping of projects in VS\" ON)\n"; 
  print $f "set_property(GLOBAL PROPERTY USE_FOLDERS \${BUILD_USE_SOLUTION_FOLDERS})\n"; 
  print $f "set(CMAKE_CXX_FLAGS $cxxflags)\n";
  print $f "set(KALDI_CXX_FLAGS \"$cxxflags\")\n";
  print $f "set(KALDI_LINKER_FLAGS $openfstlib $ldflags $openfstflags $ldlibs)\n";
  print $f "set(CMAKE_CXX_STANDARD 11)\n"; 
  foreach my $dir (@{$directories}) {
      print $f "add_subdirectory ($dir)\n";
  }
  close($f);
}

sub write_cmake {
  my $makefile = shift;
  my $s = read_whole_file($makefile);
  my $out = shift;
  
  my @path_segments = split(/[\/\\]/, $out);
  my $dir_basename = $path_segments[$#path_segments];
  #print $dir_basename;
  open(my $f, ">", "$out/CMakeLists.txt");

  if ($s =~ /[\n^]LIBNAME = ([^\n]+)\n/) {
    my $libname = $1;
    my $objfiles = "";
    my $testfiles = "";
    my $addlibs = "";

    if ($s =~ /[\n^]OBJFILES *= *([^\n]+)\n/) {
      $objfiles = $1;
    } else {
      die "Error parsing file $makefile (no OBJFILES statement)"
    }
    if ($s =~ /[\n^]TESTFILES *= *([^\n]+)\n/) {
      $testfiles = $1;
    }
    if ($s =~ /[\n^]ADDLIBS *= *([^\n]+)\n/) {
      $addlibs = $1
    }
    
    (my $ccfiles =  $objfiles) =~ s/\.o/.cc/g;
    $addlibs =~ s/\.a//g;
    $addlibs =~ s/[^ ]*\///g;
    $addlibs = join(" ", split(" ", $addlibs));
    
    $ccfiles = join(" ", split(" ", $ccfiles));
    print $f "add_library($libname STATIC $ccfiles)\n";
    print $f "target_link_libraries($libname \${KALDI_LINKER_FLAGS} $addlibs)\n" if $addlibs;
    print $f "set_target_properties($libname PROPERTIES FOLDER $dir_basename)\n";
    #print $libname . "\n";
    #print $objfiles . "\n";
    #print $testfiles . "\n";
    #print $addlibs . "\n";
  }

  if ($s =~ /[\n^]BINFILES = ([^\n]+)\n/) {
    my $binfiles = $1;
    my $testfiles = "";
    my $addlibs = "";

    if ($s =~ /[\n^]TESTFILES *= *([^\n]+)\n/) {
      $testfiles = $1;
    }
    if ($s =~ /[\n^]ADDLIBS *= *([^\n]+)\n/) {
      $addlibs = $1
    }
    $addlibs =~ s/\.a//g;
    $addlibs =~ s/[^ ]*\///g;
    $addlibs = join(" ", split(" ", $addlibs));
    
    my @binaries = split(" ", $binfiles);
    foreach my $bin (@binaries) {
      print $f "add_executable($bin $bin.cc)\n";
      print $f "target_link_libraries($bin \${KALDI_LINKER_FLAGS} $addlibs)\n" if $addlibs;
      print $f "set_target_properties($bin PROPERTIES FOLDER $dir_basename)\n";
    }
    #print $binfiles . "\n";
    #print $testfiles . "\n";
    #print $addlibs . "\n";
  }
  close($f);
}


#my $makefile = "utils/Makefile";
#my $makefiles = `find . -maxdepth 2 -mindepth 2 -type f -name Makefile -not -ipath "*build*" -not -ipath "*gst*" -not -ipath "kaldi-online" | xargs -n1 dirname | paste -s`;
#@dirs = split " ", $makefiles;

#my @dirs = ("./base", "./bin", "./chain",
#          "./chainbin", "./cudamatrix", 
#          "./decoder", "./feat", "./featbin",
#          "./fgmmbin", "./fstbin", "./fstext",
#          "./gmm", "./gmmbin", "./hmm", 
#          "./ivector", "./ivectorbin", "./kws",
#          "./kwsbin", "./lat", "./latbin", "./lm",
#          "./lmbin", "./matrix", "./nnet", 
#          "./nnet2", "./nnet2bin", "./nnet3",
#          "./nnet3bin", "./nnetbin", "./online",
#          "./online2", "./online2bin", "./onlinebin",
#          "./sgmm2", "./sgmm2bin", "./transform",
#          "./tree", "./util");
          
          
my @dirs = ("./base", "./matrix");
             
my $kaldimk = read_whole_file("kaldi.mk");
   

    
write_root_cmake($kaldimk, \@dirs, "./");
foreach my $dir (@dirs) {
  my $makefile = "$dir/Makefile";
  write_cmake($makefile, $dir);
}

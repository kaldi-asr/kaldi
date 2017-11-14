#!/usr/bin/env perl
use strict;
use warnings;

use FindBin;
use lib "$FindBin::RealBin/optimize/";
use OptimizeParams qw(&powell &easybrent &easydbrent &zbrent);
use Data::Dumper;
use Scalar::Util qw(looks_like_number);

use 5.010;

my @cmd_array = ();
my %opts = ();
my $output_dir = "";
my $result_regexp = "(.*)";
my $cmd;
my $ftol = 3e-2;
my $iftol = 1e-1;

while (@ARGV) {
  my $parm = shift @ARGV;
  if ($parm  eq "--var") {
    my $var = shift;
    die "$0: The variable $var does not contain starting value" unless $var =~ /.*=.*/;
    my @F = split "=", $var;
    die "$0: The variable $var has more than one assignments" unless @F == 2;
    die "$0: Multiple varable $F[0] definition" if defined $opts{$F[0]};
    $opts{$F[0]} = $F[1];
  } elsif ($parm eq "--output-dir") {
    $output_dir = shift;
  } elsif ($parm eq "--ftol") {
    $ftol = shift;
    die "$0: ftol parameter has to be a floating-point number" unless looks_like_number($ftol);
  } elsif ($parm eq "--iftol") {
    $iftol = shift;
    die "$0: iftol parameter has to be a floating-point number" unless looks_like_number($ftol);
  } elsif ($parm eq "--result-regexp") {
    $result_regexp = shift;
  } else {
    push @cmd_array, $parm;
    while (@ARGV) {
      push @cmd_array, shift @ARGV;
    }
  }

}


sub substitute {
  my $cmd_proto = $_[0];
  my %valhash = %{$_[1]};


  my $cmd_out = $cmd_proto;

  foreach my $key (keys %valhash) {
    #print $elem . "($key, " . $valhash{$key}. ")->";
    my $prev_cmd_out = $cmd_out;
    $cmd_out =~ s/\b$key\b/$valhash{$key}/g;
    die "$0: The variable $key is not used in the command." if $prev_cmd_out eq $cmd_out;
    #print $elem . "\n";
  }

  return $cmd_out;
}

sub f {
  state $iter = 0;
  my @params = @_;
  my $i = 0;

  my %curr_opts;
  foreach my $v (sort keys %opts) {
      $curr_opts{$v} = abs($params[$i]);
      $i += 1;
  }

  my $result;
  my $k = join(" ", substitute( $cmd, \%curr_opts));
  print "$0: Debug: $k\n";
  open(my $fh, '-|', "(set -e -o pipefail; $k) 2>&1") or die $!;
  while (my $line=<$fh>) {
    print $line;
    chomp $line;
    if ($line =~ /$result_regexp/) {
      print "$0: Line $line matches the regexp \"$result_regexp\"\n";
      $result = $line;
      $result =~ s/$result_regexp/$1/g;
    }
  }
  close($fh) or die "$0: The command didn't finish successfully: $!\n";

  my $exit = $? >> 8;
  if ( $exit != 0) {
    die "$0: The command return status indicates failure: $exit\n";
  }

  if (not defined $result) {
    die "$0: Matching the regexp on the  command output regexp didn't yield any results";
  }
  print "$0: Iteration $iter: " . join(" ", "[", @params, "] =>",  $result) . "\n";

  $iter += 1;
  return -1.0 * $result+0.0;
}


print "$0: Optimizing with " . join(" ", %opts) . "\n";
#print Dumper(\@cmd_array);

$cmd = join(" ", @cmd_array);

die "$0: Empty command \"$cmd\"" unless $cmd;
die "$0: Empty command \"$cmd\"" if $cmd =~ /^\s*$/;

my @params;
foreach my $key (sort keys %opts) {
  push @params, $opts{$key};
}

#my($xvec,$fx) = (\@params, 1);
my($xvec,$fx) = powell(\&f,\@params, $ftol, $iftol);
print "$0: Optimization finished with: " . join(" ",@$xvec, -$fx), "\n";


@params=@{$xvec};
foreach my $v (sort keys %opts) {
    $opts{$v} = abs(shift @params);
}
$cmd=substitute($cmd, \%opts);

{
  open(my $param_file, "> $output_dir/params") || die "Cannot open file $output_dir/params: $!";
  print $param_file "$_=$opts{$_}\n" for (sort keys %opts);
  print $param_file "criterion=", -$fx;
  close($param_file);
}

{
  open(my $param_file, "> $output_dir/command.sh");
  print $param_file "$cmd\n";
  close($param_file);
}

{
  open(my $param_file, "> $output_dir/params.sh");
  print $param_file "declare -A params;\n";
  print $param_file "params[$_]=$opts{$_}\n" for (sort keys %opts);
  close($param_file);
}


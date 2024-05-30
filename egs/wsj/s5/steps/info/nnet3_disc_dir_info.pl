#!/usr/bin/perl -w

use Fcntl;

# we may at some point support options.

$debug = 0;  # we set it to 1 for debugging the script itself.

if ($ARGV[0] eq "--debug") {
  $debug = 1;
  shift @ARGV;
}

if (@ARGV == 0) {
  print STDERR "Usage: steps/info/nnet3_disc_dir_info.pl [--debug] <nnet3-disc-dir1> [<nnet3-disc-dir2> ... ]\n" .
               "e.g: steps/info/nnet3_dir_info.pl exp/nnet3/tdnn_sp_smbr\n" .
               "This script extracts some important information from the logs\n" .
               "and displays it on a few lines.\n" .
               "The --debug option is just to debug the script itself.\n" .
               "This program exits with status 0 if it seems like the argument\n" .
               "really was a GMM dir, and 1 otherwise.\n";
  exit(1);
}

if (@ARGV > 1) {
  # repeatedly invoke this program with each of the remaining args.
  $exit_status = 0;
  if ($debug) { $debug_opt = "--debug " } else { $debug_opt = ""; }
  foreach $dir (@ARGV) {
    if (system("$0 $debug_opt$dir") != 0) {
      $exit_status = 1;
    }
  }
  exit($exit_status);
}

# from this point we can assume we're invoked with one argument.
$nnet_dir = shift @ARGV;

# This function returns an array of iteration numbers, one
# for each epoch that has already completed (but including
# epoch zero)... e.g.
# it might return (0, 194, 388, 582).
# This is done by reading the soft links, e.g. epoch1.mdl ->194.mdl
sub get_iters_for_epochs {
  my @ans = ();
  for (my $n = 0; 1; $n++) {
    if (-l "$nnet_dir/epoch$n.mdl") {
      my $link_name = readlink("$nnet_dir/epoch$n.mdl");
      if ($link_name =~ m/^(\d+).mdl/) {
        my $iter = $1;
        push @ans, $iter;
      } else {
        die "unexpected link name $nnet_dir/epoch$n.mdl -> $link_name";
      }
    } else {
      if (@ans == 0) {
        die "$nnet_dir does not seem to be a discriminative-training dir " .
          "(expected a link $nnet_dir/epoch0.mdl)";
      }
      return @ans;
    }
  }
}


sub get_num_jobs {
  my $j = 1;
  for (my $j = 1; 1; $j++) {
    if (! -f "$nnet_dir/log/train.0.$j.log") {
      if ($j == 1) {
        die "$nnet_dir does not seem to be a discriminative-training dir " .
          "(expected $nnet_dir/log/train.0.1.log to exist)";
      } else {
        return $j - 1;
      }
    }
  }
}

# returns a string describing the effective learning rate and possibly
# any final-layer-factor.
sub get_effective_learning_rate_str {
  # effective learning rate is the actual learning rate divided by the
  # number of jobs.
  my $convert_log = "$nnet_dir/log/convert.log";
  if (-f $convert_log) {
    open(F, "<$convert_log");
    while (<F>) {
      if (m/--edits/) {
        if (m/set-learning-rate learning-rate=(\S+); set-learning-rate name=output.affine learning-rate=([^"']+)["']/) {
          my $learning_rate = $1;
          my $last_layer_factor = sprintf("%.2f", $2 / $1);
          my $num_jobs = get_num_jobs();
          my $effective_learning_rate = sprintf("%.3g", $learning_rate / $num_jobs);
          close(F);
          return "effective-lrate=$effective_learning_rate;last-layer-factor=$last_layer_factor";
        } elsif (m/set-learning-rate learning-rate=([^"']+)["']/) {
          my $learning_rate = $1;
          my $num_jobs = get_num_jobs();
          my $effective_learning_rate = sprintf("%.3g", $learning_rate / $num_jobs);
          close(F);
          return "effective-lrate=$effective_learning_rate";
        }
      }
    }
  } else {
    die("Expected file $convert_log to exist");
  }
  close(F);
  return "lrate=??";  # could not parse it from the log.
}


# prints some info about the objective function...
sub get_objf_str {
  my @iters_for_epochs = get_iters_for_epochs();
  if (@iters_for_epochs == 1) {
    die("No epochs have finished in directory $nnet_dir")
  }
  # will produce output like:
  # iters-per-epoch=123;epoch[0,1,2,3,4]:train-objf=[0.89,0.92,0.93,0.94],valid-objf=[...],train-counts=[...],valid-counts=[...]"
  # the "counts" are the average num+den occupation counts in the lattices; it's a measure of how much confusability
  # there still is in the lattices.
  my $iters_per_epoch = $iters_for_epochs[1] - $iters_for_epochs[0];
  my $ans = "iters-per-epoch=$iters_per_epoch";
  $ans .= ";epoch[" . join(",", 0..$#iters_for_epochs) . "]:";
  my @train_objfs = ();
  my @train_counts = ();
  my @valid_objfs = ();
  my @valid_counts = ();
  foreach $iter (@iters_for_epochs) {
    if ($iter > 0) { $iter -= 1; }  # last iter will not exist.
    my $train_log = "$nnet_dir/log/compute_objf_train.$iter.log";
    my $valid_log = "$nnet_dir/log/compute_objf_valid.$iter.log";
    if (!open (T, "<$train_log")){  print STDERR "$0: warning: Expected file $train_log to exist\n"; }
    if (!open (V, "<$valid_log")){  print STDERR "$0: warning: Expected file $valid_log to exist\n"; }
    my $train_count = "??";
    my $valid_count = "??";
    my $train_objf = "??";
    my $valid_objf = "??";
    while (<T>) {
      if (m/num\+den count.+is (\S+) per frame/) { $train_count = sprintf("%.2f", $1); }
      if (m/Overall.+ is (\S+) per frame/) { $train_objf = sprintf("%.2f", $1); }
    }
    close(T);
    while (<V>) {
      if (m/num\+den count.+is (\S+) per frame/) { $valid_count = sprintf("%.2f", $1); }
      if (m/Overall.+ is (\S+) per frame/) { $valid_objf = sprintf("%.2f", $1); }
    }
    push @train_objfs, $train_objf;
    push @train_counts, $train_count;
    push @valid_objfs, $valid_objf;
    push @valid_counts, $valid_count;
    close(V);
  }
  $ans .= "train-objf=[" . join(",", @train_objfs) .
       "],valid-objf=[" . join(",", @valid_objfs) .
       "],train-counts=[" . join(",", @train_counts) .
       "],valid-counts=[" . join(",", @valid_counts) . "]";
  return $ans;
}




$output_string = "$nnet_dir:num-jobs=".get_num_jobs().";" .
     get_effective_learning_rate_str() . ";" . get_objf_str();

print "$output_string\n";

exit(0);

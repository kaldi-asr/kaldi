#!/usr/bin/perl -w

use Fcntl;

# we may at some point support options.

$debug = 0;  # we set it to 1 for debugging the script itself.

if ($ARGV[0] eq "--debug") {
  $debug = 1;
  shift @ARGV;
}

if (@ARGV == 0) {
  print STDERR "Usage: steps/info/nnet2_dir_info.pl [--debug] <nnet3-dir1> [<nnet3-dir2> ... ]\n" .
               "e.g: steps/info/nnet2_dir_info.pl exp/nnet3/tdnn_sp\n" .
               "This script extracts some important information from the logs\n" .
               "and displays it on a single (rather long) line.\n" .
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

$nnet_dir = shift @ARGV;

sub list_all_log_files {
  my @ans = ();
  my $dh;
  if (!opendir($dh, "$nnet_dir/log")) { return (); }
  @ans = readdir $dh;
  closedir $dh;
  return @ans;
}


# returns 1 if the diagnostics are finished on this iter, else 0.
sub diagnostics_are_finished_on_iter {
  my $ans = 1;
  my $iter = shift @_;
  if (!open(F, "<$nnet_dir/log/compute_prob_train.$iter.log")) {
    return 0;
  }
  $found_loglike = 0;
  while (<F>) {
    if (m/Overall log-likelihood/) { $found_loglike = 1; }
  }
  if (!$found_loglike) { $ans = 0; }
  close(F);
  if (!open(F, "<$nnet_dir/log/compute_prob_valid.$iter.log")) {
    return 0;
  }
  $found_loglike = 0;
  while (<F>) {
    if (m/Overall log-likelihood/) { $found_loglike = 1; }
  }
  if (!$found_loglike) { $ans = 0; }
  close(F);
  return $ans;
}

# get the number of iterations.
# note: the iterations go from 0 to num-iters-1.
# if num_iters = 0 this program will just exit with status 1.
# we may return a number slightly less than the number of iterations
# in order to ensure that the compute_prob_train and compute_prob_valid
# processes have finished.
sub get_num_iters {
  my $iter = 0;
  while (defined $log_file_hash{"train.$iter.1.log"}) {
    $iter++;
  }
  if ($iter == 0) {
    die "$nnet_dir does not seem to be an nnet3 neural net training directory.";
  }
  my $last_iter = $iter - 1;
  # find an iteration where the diagnostic jobs compute_prob_{train,valid}.$last_iter.log are done.
  for (my $chosen_last_iter = $last_iter;
       $chosen_last_iter >= $last_iter - 6 && $chosen_last_iter >= 0;
       $chosen_last_iter--) {
    if (! diagnostics_are_finished_on_iter($chosen_last_iter)) {
      if ($debug) {
        print STDERR "nnet3_dir_info.pl: diagnostics not finished running on iteration $chosen_last_iter\n";
      }
    } else {
      return $chosen_last_iter + 1;
    }
  }
  # OK, something's not right, just return the original iteration.
  return $iter;
}

sub get_num_jobs_initial {
  my $num_jobs = 1;
  while (defined $log_file_hash{"train.0.$num_jobs.log"}) {
    $num_jobs++;
  }
  $num_jobs--;
  if ($num_jobs == 0) {
    die "$nnet_dir does not seem to be an nnet3 neural net training directory.";
  }
  return $num_jobs;
}


sub get_num_jobs_final {  # expects $num_iters to exist as a global variable.
  my $final_iter = $num_iters - 1;
  my $num_jobs = 1;
  while (defined $log_file_hash{"train.$final_iter.$num_jobs.log"}) {
    $num_jobs++;
  }
  $num_jobs--;
  if ($num_jobs == 0) {
    die "$nnet_dir does not seem to be an nnet3 neural net training directory.";
  }
  return $num_jobs;
}

sub get_combine_info {
  # returns a string with info about the combination stage, or the empty
  # string if there wasn't one.
  if (defined $log_file_hash{"combine.log"} &&
      open(F, "<$nnet_dir/log/combine.log")) {
    while (<F>) {
      if (m/Combining nnets, objective function changed from (\S+) to (\S+)/) {
        close(F);
        return sprintf(" combine=%.2f->%.2f", $1, $2);
      }
    }
  }
  return "";
}

# this is used in get_loglike_and_accuracy to format
# strings like ' loglike[32,48,final],train/valid=(-2.43,-2.32,-2.21/-2.84,-2.71,-2.68)'.
sub get_printed_string {
  # $name might be 'loglike', for example.
  my ($name, $iters_array_ref, $train_hash_ref, $valid_hash_ref) = @_;
  my @iters_array = @$iters_array_ref;
  my %train_hash = %$train_hash_ref;  # hash from iter-string to value.
  my %valid_hash = %$valid_hash_ref;  # hash from iter-string to value.
  my @iters_to_print = ();
  my @train_values_to_print = ();
  my @valid_values_to_print = ();
  foreach my $iter (@iters_array) {
    if (defined($train_hash{$iter}) && defined($valid_hash{$iter})) {
      push @iters_to_print, $iter;
      push @train_values_to_print, sprintf("%.2f", $train_hash{$iter});
      push @valid_values_to_print, sprintf("%.2f", $valid_hash{$iter});
    }
  }
  if (@iters_to_print == 0) {  return ""; }
  my $joined_iters = join(",", @iters_to_print);
  my $joined_train_values = join(",", @train_values_to_print);
  my $joined_valid_values = join(",", @valid_values_to_print);
  return " ${name}:train/valid[$joined_iters]=($joined_train_values/$joined_valid_values)";
}

# invoke this as get_objf_iter($iter1, $iter2,..) where $iterN is the string-valued
# iteration, e.g. "92", or "final", or "combined", such that we expect
# $nnet_dir/log/compute_prob_{train,valid}.$iterN.log to exist.
sub get_loglike_and_accuracy_info {
  my @iters_array = @_;
  my %iter_to_train_loglike = ();
  my %iter_to_valid_loglike = ();
  my %iter_to_train_accuracy = ();
  my %iter_to_valid_accuracy = ();


  foreach my $iter (@iters_array) {
    if (defined $log_file_hash{"compute_prob_train.$iter.log"} &&
        defined $log_file_hash{"compute_prob_valid.$iter.log"} &&
        open(F, "<$nnet_dir/log/compute_prob_train.$iter.log") &&
        open(G, "<$nnet_dir/log/compute_prob_valid.$iter.log")) {
      while (<F>) {
        if (m/average probability is (\S+) and accuracy is (\S+) with total weight \S+/) {
          $iter_to_train_loglike{$iter} = $1;
          $iter_to_train_accuracy{$iter} = $2;
        }
      }
      close(F);
      while (<G>) {
        if (m/average probability is (\S+) and accuracy is (\S+) with total weight \S+/) {
          $iter_to_valid_loglike{$iter} = $1;
          $iter_to_valid_accuracy{$iter} = $2;
        }
      }
      close(G);
    }
  }
  $ans = "";
  $ans .= get_printed_string("loglike", \@iters_array, \%iter_to_train_loglike,
                             \%iter_to_valid_loglike);
  $ans .= get_printed_string("accuracy", \@iters_array, \%iter_to_train_accuracy,
                             \%iter_to_valid_accuracy);
  return $ans;
}

# invoke this as get_progress_info($iter), e.g. set $iter to the last
# iteration number.
sub get_progress_info {
  my $iter = shift @_;
  if (!defined $log_file_hash{"progress.$iter.log"} ||
      !open(F, "<$nnet_dir/log/progress.$iter.log")) {
    return "";
  }
  my $num_parameters = "0";
  my $output_dim = 0;
  my $input_dim = 0;
  while (<F>) {
    if (m/^parameter-dim (\S+)/) {
      $num_parameters = sprintf("%.1fM", $1 / 1000000.0);
    }
    if (m/^input-dim (\S+)/) {
      $input_dim = $1;
    }
    if (m/^output-dim (\S+)/) {
      $output_dim = $1;
    }
  }
  close(F);
  $ans = "";
  if ($num_parameters ne "0") {  $ans .= " num-params=$num_parameters"; }
  if ($output_dim > 0 && $input_dim > 0) {
    $ans .= " dim=$input_dim->$output_dim";
  } elsif ($output_dim > 0) {
    $ans .= " output-dim=$output_dim";
  }
  return $ans;
}

# return 1 if we seem to have finished training, else 0.
sub finished_training {
  return defined $log_file_hash{"compute_prob_train.final.log"} ||
    defined $log_file_hash{"compute_prob_train.combined.log"};
}

@log_files = list_all_log_files();
if (@log_files == 0) {  exit(1); }
$log_file_hash = ();
foreach $f (@log_files) { $log_file_hash{$f} = 1; }

$num_iters = get_num_iters();
$num_jobs_initial = get_num_jobs_initial();
$num_jobs_final = get_num_jobs_final();
$last_iter = $num_iters - 1;
$two_thirds_iter = int($last_iter * 0.666);

$output_string = "$nnet_dir: num-iters=$num_iters";

$output_string .= " nj=$num_jobs_initial..$num_jobs_final";

$output_string .= get_progress_info("$last_iter");

$output_string .= get_combine_info();



# note: IIRC some of the scripts use the name 'combined' for the model after
# combination, and some 'final', so we try both; only one of these will
# actually produce any output.


@iters_array = ("$two_thirds_iter", "$last_iter", "final", "combined");

$output_string .= get_loglike_and_accuracy_info(@iters_array);

print "$output_string\n";

exit(0);

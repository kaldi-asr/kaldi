#!/usr/bin/perl -w

use Fcntl;

# we may at some point support options.

$debug = 0;  # we set it to 1 for debugging the script itself.

if ($ARGV[0] eq "--debug") {
  $debug = 1;
  shift @ARGV;
}

if (@ARGV == 0) {
  print STDERR "Usage: steps/info/gmm_dir_info.pl [--debug] <gmm-dir1> [<gmm-dir2> ... ]\n" .
               "e.g: steps/info/gmm_dir_info.pl exp/tri3\n" .
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


$gmm_dir = shift @ARGV;

sub list_all_log_files {
  my @ans = ();
  my $dh;
  if (!opendir($dh, "$gmm_dir/log")) { return (); }
  @ans = readdir $dh;
  closedir $dh;
  return @ans;
}


sub get_num_jobs {
  if (! -d $gmm_dir) {
    print STDERR "steps/info/gmm_dir_info.pl: no such directory $gmm_dir\n";
    exit(1);
  }
  if (!open(F, "<$gmm_dir/num_jobs")) {
    print STDERR "steps/info/gmm_dir_info.pl: no such file $gmm_dir/num_jobs\n";
  }
  my $num_jobs = <F>;
  if (!($num_jobs > 0)) {
    print STDERR "steps/info/gmm_dir_info.pl: bad contents of file $gmm_dir/num_jobs\n";
  }
  close(F);
  return 0 + $num_jobs;  # force conversion to integer.
}

# this function returns a string containing info from the last set of alignment
# jobs.  it may be empty if no alignment info was found, or if it didn't have the
# expected contents.
sub get_last_align_info {
  $max_align_iter = -1;
  foreach $f (@log_files) {
    if ($f =~ m:^align.(\d+).1.log$: && $1 > $max_align_iter) {
      $max_align_iter = $1;
    }
  }
  if ($debug) {
    print STDERR "max-align-iter=$max_align_iter\n";
  }
  if ($max_align_iter == -1) { return ""; }  # something went wrong; return no info.

  $num_utts = 0;
  $num_utts_err = 0;
  $num_utts_retry = 0;
  $num_frames = 0;
  $tot_loglike = 0;
  if ($debug) {
    print STDERR "Starting reading alignment logs\n";
  }
  for ($j = 1; $j <= $num_jobs; $j++) {
    if (open(F, "${gmm_dir}/log/align.$max_align_iter.$j.log")) {
      # we only need the last few lines of the file, e.g. the last 5 lines which
      # would normally be about 400 characters... so the last 1000 characters
      # should be enough.
      seek(F, Fcntl::SEEK_END, -1000);
      while (<F>) {
        if (m/Overall log-likelihood per frame is (\S+) over (\S+) frames./) {
          $tot_loglike += $1 * $2;
          $num_frames += $2;
        } elsif (m/Retried (\S+) out of (\S+) utterances/) {
          $num_utts_retry += $1;
          $num_utts += $2;
        } elsif (m/Done \S+, errors on (\S+)/) {
          $num_utts_err += $1;
        }
      }
      close(F);
    }
  }
  if ($debug) {
    print STDERR "Done reading alignment logs\n";
  }
  if ($num_utts == 0 || $num_frames == 0) { return ""; }  # something went wrong.

  # note: the number of hours of data, e.g. "3.23h data", assumes 100 frames
  # per second, which is almost always true for GMM-based systems.
  return sprintf(" align prob=%.2f over %.2fh [retry=%.1f%%, fail=%.1f%%]",
                 ($tot_loglike / $num_frames), ($num_frames / 360000.0),
                 ($num_utts_retry * 100.0 / $num_utts), ($num_utts_err * 100.0 / $num_utts));
}


# this function returns a string containing info from the last update
# job.  Right now it includes info about the num-states and num-gauss
# and the percentage of Gaussians that had variances floored; we
# also say how much data was used if this
# the string may be empty if no such job was found.
sub get_last_update_info {
  $max_update_iter = -1;
  foreach $f (@log_files) {
    if ($f =~ m:^update.(\d+).log$: && $1 > $max_update_iter) {
      $max_update_iter = $1;
    }
  }
  if ($debug) {
    print STDERR "max-update-iter=$max_update_iter\n";
  }
  if ($max_update_iter == -1) { return ""; }  # something went wrong; return no info.


  $num_gauss = 0;
  $num_gauss_floored = 0;  # number of Gaussians with at least one variance floored.
  $num_gauss_removed = 0;  # number of Gaussians removed due to low-counts.
  $num_gauss_tot = 0;     # total number of Gaussians before splitting.
  $num_gauss_after_split = 0;  # total number of Gaussians after splitting [will
                               # usually be same as before, on last iter.]
  $num_states = 0;  # total number of states [pdf-ids]
  $num_frames = 0;  # total number of frames.
  $loglike = 0;  # log-likelihood [from auxf].

  if (open(F, "<${gmm_dir}/log/update.$max_update_iter.log")) {
    while (<F>) {
      if (m/variance elements floored in (\S+) Gaussians, out of (\S+)/) {
        $num_gauss_floored = $1;
        $num_gauss_tot = $2;
      } elsif (m/Overall avg like per frame = (\S+) over (\S+) frames/) {
        $loglike = $1;
        $num_frames = $2;
      } elsif (m/Split (\S+) states .+ split #Gauss from \S+ to (\S+)/) {
        $num_states = $1;
        $num_gauss_after_split = $2;
      }
    }
    close(F);
  } else {
    return "";  # something went wrong.
  }
  $ans = "";

  if (($align_info eq "" || ! defined $align_info)) {
    # add some info that we'd otherwise get from the alignment jobs.
    if ($num_frames != 0) {
      # add info about how much data we trained on.
      $ans .= sprintf(" %.2fh data", $num_frames / 360000.0);
    }
    if ($loglike != 0) {
      $ans .= sprintf(" log-like=%.2f", $loglike);
    }
  }

  if ($num_states != 0) {
    $ans .= sprintf(" states=%d", $num_states);
  }

  # the next line is really just in case there was no splitting done-- in that
  # case we get the num-gauss from the line about the variance flooring.
  $max_num_gauss = ($num_gauss > $num_gauss_after_split ? $num_gauss : $num_gauss_after_split);
  if ($max_num_gauss > 0) { $ans .= " gauss=$max_num_gauss"; }

  if ($num_gauss > 0 && $num_gauss_removed > 0) {
    $ans .= sprintf(" lowcount-gauss-removed=%d", $num_gauss_removed);
  }

  if ($num_gauss > 0 && $num_gauss_floored > 0) {
    $ans .= sprintf(" gauss-floored=%.02%%", $num_gauss_floored * 100.0 / $num_gauss);
  }
  return $ans;
}


sub get_fmllr_info {
  my %fmllr_num_frames = ();  # maps from fmllr iteration to num-frames
  my %fmllr_auxf_impr = ();  # maps from fmllr iteration to total auxf impr times num-frames.
  foreach $log_file (@log_files) {
    if ($log_file =~ m/^fmllr.(\d+).(\d+).log$/) {
      $iter = $1;
      $job_number = $2;
      if ($job_number <= $num_jobs && open(F, "<$gmm_dir/log/$log_file")) {
        while (<F>) {
          if (m/Overall fMLLR auxf impr per frame is (\S+) over (\S+) frames/) {
            $fmllr_num_frames{$iter} += $2;
            $fmllr_auxf_impr{$iter} += $1 * $2;
          }
        }
        close(F);
      }
    }
  }
  my $tot_auxf_impr = 0.0;
  my $num_frames = 0.0;
  # the fMLLR auxf impr will be summed over the fMLLR iterations.
  foreach $iter (sort(keys %fmllr_auxf_impr)) {
    if ($debug) {
      print STDERR "fmllr iter $iter: $fmllr_auxf_impr{$iter} / $fmllr_num_frames{$iter}\n";
    }
    $tot_auxf_impr += $fmllr_auxf_impr{$iter} / $fmllr_num_frames{$iter};
    $num_frames = $fmllr_num_frames{$iter};  # take the num-frames from the final iteration.
  }
  if ($tot_auxf_impr != 0.0 && $num_frames != 0.0) {
    return sprintf(" fmllr-impr=%.2f over %.2fh", $tot_auxf_impr, $num_frames / 360000.0);
  } else {
    return "";
  }
}

sub get_mllt_info {
  # note: both the objective improvement and logdet are summed over
  # all the iterations of MLLT update.
  my $mllt_objf_impr = 0.0;
  my $mllt_logdet = 0.0;

  foreach $log_file (@log_files) {
    if ($log_file =~ m/^mupdate.\d+.log$/) {
      if (open(F, "<$gmm_dir/log/$log_file")) {
        while (<F>) {
          if (m/Overall objective function improvement for MLLT is (\S+) over \S+ frames, logdet is (\S+)/) {
            $mllt_objf_impr += $1;
            $mllt_logdet += $2;
          }
        }
        close(F);
      }
    }
  }
  if ($mllt_objf_impr != 0.0 && $mllt_logdet != 0.0) {
    return sprintf(" mllt:impr,logdet=%.2f,%.2f", $mllt_objf_impr, $mllt_logdet);
  } else {
    return "";
  }
}

sub get_tree_info {
  $ans = "";
  if (open(F, "<$gmm_dir/log/build_tree.log")) {
    while (<F>) {
      if (m/Including just phones that were split, improvement is (\S+) per frame/) {
        $ans = sprintf(" tree-impr=%.2f", $1);
      }
    }
    close(F);
  }
  return $ans;
}

sub get_lda_info {
  $ans = "";
  if (open(F, "<$gmm_dir/log/lda_est.log")) {
    while (<F>) {
      if (m/Sum of selected singular values is (\S+)/) {
        $ans = sprintf(" lda-sum=%.2f", $1);
      }
    }
    close(F);
  }
  return $ans;
}


@log_files = list_all_log_files();

if (@log_files == 0) {
  exit(1);
}

$output_string = "$gmm_dir:";

$num_jobs = get_num_jobs();  # will crash on failure.

$output_string .= " nj=$num_jobs";

$insufficient_output_string = $output_string;

$align_info =  get_last_align_info();
$output_string .= $align_info;

$output_string .= get_last_update_info($align_info);

$output_string .= get_fmllr_info();

$output_string .= get_tree_info();

$output_string .= get_lda_info();

$output_string .= get_mllt_info();

print $output_string . "\n";

if ($output_string eq $insufficient_output_string) {
  # if we only had "$gmm_dir: nj=$num_jobs", then it's probably not a GMM dir:
  # exit with status 1.
  exit(1);
}

exit(0);


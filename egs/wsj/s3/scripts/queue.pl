#!/usr/bin/perl
use File::Basename;
use Cwd;

# queue.pl has the same functionality as run.pl, except that
# it runs the job in question on the queue.  
# Suppose from the directory /cur/location, you run the command:
#  queue.pl somedir/some.log my-prog "--opt=foo bar" foo \|  other-prog baz
# and queue.pl will do something roughly equivalent to the following:
#
# cat >somedir/q/some.sh <<EOF
# #!/bin/bash
# cd /cur/location
# . path.sh
# ( my-prog '--opt=foo bar' foo |  other-prog baz ) >& some.log
# EOF
# qsub -sync y -j y -o /cur/location/some.log /cur/location/q/some.sh && exit 0;
# qsub -sync y -j y -o /cur/location/some.log /cur/location/q/some.sh 
#
# What this means is it creates a .sh file to put the command in,
# along with changing the directory and setting the path.  It then runs
# this command on the queue, and tries a second time if it fails the first
# time.
#
# Note: as a special case, if the log file is in a directory called log/,
# this program puts the q/ directory one level up.
#
# As a mechanism to pass options to qsub: this program will take
# any leading arguments beginning with "-", and the ones immediately 
# following them, and give them to qsub.  E.g. if you call
#  queue.pl -l ram_free=600M,mem_free=600M foo bar
# it will pass the first two arguments to qsub.

$qsub_opts = "";
while ($ARGV[0] =~ m:^-:) {
    $qsub_opts .= (shift @ARGV) . " ";
    if (@ARGV == 0) { 
        die "Invalid command-line arguments to queue.pl: expecting a next argument to qsub."; 
    }
    $qsub_opts .= (shift @ARGV) . " ";
}

@ARGV < 2 && die "usage: run.pl log-file command-line arguments...";
$cwd = getcwd;
$logfile = shift @ARGV;
if ($logfile !~ m:^/:) { $logfile = "$cwd/$logfile"; }

#
# Work out the command; quote escaping is done here.
#
$cmd = "";

foreach $x (@ARGV) { 
    if ($x =~ m/^\S+$/) { $cmd .=  $x . " "; }
    elsif ($x =~ m:\":) { $cmd .= "'\''$x'\'' "; }
    else { $cmd .= "\"$x\" "; } 
}

#
# Work out the location of the script file, and open it for writing.
#
$cwd = getcwd;
$dir = dirname($logfile);

# go one level up if log file is in a directory named log/
if (basename($dir) eq "log") { $dir = dirname($dir); }
$dir = "$dir/q";
if (!-d $dir) { system "mkdir $dir 2>/dev/null"; } # another job may be doing this...
$base = basename($logfile);
# Replace trailing .log with .sh
$base =~ s:\.[a-z]+$:.sh: || die "Could not make sense of log-file name (expect a suffix e.g. .log): $logfile";
$shfile = "$dir/$base";
open(S, ">$shfile") || die "Could not write to script file $shfile";
`chmod +x $shfile`;

$qsub_cmd = "qsub -sync y -j y -o $logfile $qsub_opts $shfile >>$dir/queue.log 2>&1";
#
# Write to the script file, and close it.
#
print S "#!/bin/bash\n";
print S "cd $cwd\n";
print S ". path.sh\n";
print S "echo Running on \`hostname\` >$logfile\n";
print S "echo Started at \`date\` >>$logfile\n";
print S " ( $cmd ) 2>>$logfile >>$logfile\n";
print S "ret=\$?\n";
print S "echo Finished at \`date\` >>$logfile\n";
print S "exit \$ret\n";
print S "## submitted with:\n";
print S "# $qsub_cmd\n";
close(S) || die "Could not close script file $shfile";

#
# Try to run the script file, on the queue.
#
system "$qsub_cmd";
if ($? == 0) { exit(0); }
$errmsgs = `cat $dir/queue.log`;
if ($errmsgs =~ m/containes/) { # the error message "range_list containes no elements"
  # seems to be encountered due to a bug in grid engine... since this appears to be 
  # intermittent, we try a bunch of times, with sleeps in between, if this happens.
  print STDERR "Command writing to $logfile failed, apparently due to queue bug " .
      " (range_list containes no elements)... will try again a few times.\n";
  $delay = 60; # one minute delay initially.
  for ($x = 1; $x < 10; $x++) {
      print STDERR "[$x/10]";
      sleep($delay);
      $delay += 60*5; # Add 5 minutes to the delay.
      system "$qsub_cmd";
      if ($? == 0) { exit(0); }
  }
}

print STDERR "Command writing to $logfile failed; trying again\n";
system "mv $logfile $logfile.bak";
system "$qsub_cmd";
if ($? == 0) { 
    exit(0); 
} else {
    print STDERR "Command writing to $logfile failed second time.  Command is in $shfile\n";
    exit(1);
}

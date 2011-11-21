#!/usr/bin/perl
use File::Basename;
use Cwd;
use Time::HiRes qw (usleep);

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
# qrsh -j y -now no -o /cur/location/some.log /cur/location/q/some.sh  && exit 0;
# qrsh -j y -now no -o /cur/location/some.log /cur/location/q/some.sh 
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

$qsub_cmd = "qrsh -now no $qsub_opts $shfile";
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

$num_tries = 2; # Unless we fail with exit status 0 (i.e. the job returns with
                # exit status 0 but there is no "finished" message at the
                # end of the log-file), this is how many tries we do.  But
               # if we get this nasty exit-status-0 thing, which seems to be
               # unpredictable and relates somehow to the queue system,
               # we'll try more times and will put delays in.
$max_tries = 10;      
$delay = 1; # seconds.  We'll increase this.
$increment = 30; # increase delay by 30 secs each time.
for ($try = 1; ; $try++) {
#
# Try to run the script file, on the queue.
#
  system "$qsub_cmd";

  # Since we moved from qsub -sync y to qrsh (to work around a bug in
  # GridEngine), we have had jobs fail yet return zero exit status.
  # The "tail -1 $logfile" below is to try to catch this.
  $ret = $?;
  if ($ret == 0) { ## Check it's really successful: log-file should say "Finished" at end...
    # but sleep first, for 0.1 seconds; need to wait for file system to sync.
    usleep(100000);
    if(`tail -1 $logfile` =~ m/Finished/) { exit(0); }
    usleep(500000); # wait another half second and try again, in case file system is syncing slower than that.
    if(`tail -1 $logfile` =~ m/Finished/) { exit(0); }
    sleep(1); # now a full second.
    if(`tail -1 $logfile` =~ m/Finished/) { exit(0); }
  }
  
  if ($try < $num_tries || ($ret == 0 && $try < $max_tries)) {
    print STDERR "Command writing to $logfile failed with exit status $ret [on try $try]; waiting $delay seconds and trying again\n";
    sleep($delay);
    $delay += $increment;
    system "mv $logfile $logfile.bak";
  } else {
    print STDERR "Command writing to $logfile failed after $try tries.  Command is in $shfile\n";
    exit(1);
  }
}
